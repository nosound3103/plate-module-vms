"""PLATE-compatible evidence delivery using the existing asynchronous MQTT client."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import logging
import re
import time
from .events import build_plate_event
from .red_light_config import RedLightSettings
from .red_light_evidence import prepare_evidence, write_manifest, upload_evidence

LOGGER = logging.getLogger(__name__)


def iso_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp, timezone(timedelta(hours=7))).isoformat(timespec="milliseconds")


class PlatePublishGate:
    """Serialize same camera/plate events after PUBACK + backend dedup window.

    This covers this process only. Backend still needs event-aware idempotency
    for other publishers and arbitrary broker/backend delivery delays.
    """
    def __init__(self, publish, guard_seconds, status=None):
        self.publish, self.guard, self.status = publish, guard_seconds, status
        self.last = {}

    def __call__(self, event, *, camera_code, object_key):
        now = time.monotonic()
        plate = event["payload"]["vehicle"]["license_plate"]["text"]
        identity = (event["camera_id"], plate)
        for key, (previous_key, queued_at) in tuple(self.last.items()):
            receipt = self.status(previous_key) if self.status else ("ACK", queued_at)
            if receipt and receipt[0] != "PENDING" and now-receipt[1] >= self.guard:
                del self.last[key]
        if plate and identity in self.last:
            return False
        accepted = self.publish(event, camera_code=camera_code, object_key=object_key)
        if accepted and plate:
            self.last[identity] = (object_key, now)
        return accepted


def violation_read(violation, settings):
    """Only accept OCR tied to this continuous identity, with a real plate box."""
    track = violation.track
    proof = violation.read_at_crossing
    if proof is None:
        if track.identity_epoch != violation.identity_epoch or not track.association_reliable:
            return None
        read = track.best_read()
        proof = track.read_evidence.get(read.license_plate.text) if read else None
    if proof is None or proof.identity_epoch != violation.identity_epoch:
        return None
    read, plate = proof.read, proof.read.license_plate
    if (proof.captured_at > violation.frame_timestamp+settings.plate_after_crossing_seconds
            or plate.confidence < settings.plate_min_confidence
            or plate.detection_confidence < settings.plate_min_detection_confidence
            or plate.bbox is None or not re.sub(r"[^A-Z0-9]", "", plate.text.upper())):
        return None
    x1, y1, x2, y2 = read.vehicle.bbox
    px1, py1, px2, py2 = plate.bbox
    if not (x1 <= px1 < px2 <= x2 and y1 <= py1 < py2 <= y2):
        return None
    return proof


def build_red_light_event(camera, violation, snapshot, settings=None):
    if (violation.signal is None or violation.signal.state != "RED"
            or violation.signal.timestamp != violation.frame_timestamp
            or violation.frame_sequence <= 0 or len(violation.light_points) < 3):
        return None
    proof = violation_read(violation, settings or RedLightSettings())
    if proof is None:
        return None
    read = proof.read
    key = f"{violation.track_key}:RED_LIGHT"
    view = replace(violation.track, crossed_epoch=violation.timestamp, object_key=key,
                   box=violation.vehicle_box, vehicle=violation.vehicle,
                   votes={read.license_plate.text: (1, read.license_plate.confidence, read)},
                   direction=violation.direction)
    event = build_plate_event(camera, view)
    vehicle = event["payload"]["vehicle"]
    event["payload"]["vehicle"] = {k: v for k, v in vehicle.items() if v is not None}
    event["payload"].update({
        "violation_type_code": "RED_LIGHT",
        "violation_evidence": {
            "rule": "red_light_crossing", "confidence": round(violation.confidence, 4),
            "traffic_light_state": "RED", "from_zone": "STOP-ZONE", "to_zone": "VIOLATION-ZONE",
            "track_id": violation.track_id, "crossing_timestamp": event["event_time"],
            "snapshot_timestamp": iso_timestamp(violation.frame_timestamp),
            "frame_sequence": violation.frame_sequence,
            "vehicle_bbox": list(violation.vehicle_box),
            "traffic_light_polygon": violation.light_points,
            "traffic_light_scores": violation.signal.scores if violation.signal else {},
            "plate_frame_sequence": proof.sequence,
            "plate_timestamp": iso_timestamp(proof.captured_at),
            "plate_vehicle_bbox": read.vehicle.bbox,
            "plate_bbox": read.license_plate.bbox,
            "identity_epoch": violation.identity_epoch,
        },
    })
    event["snapshot_url"] = snapshot
    return event


class RedLightEventPublisher:
    """Bounded pending OCR, one background disk worker, nonblocking publish/retry."""
    def __init__(self, settings, publish, *, snapshots=None):
        self.settings, self.publish = settings, publish
        self.snapshots = snapshots
        self.pending = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="red-light-evidence")

    def submit(self, camera, violation):
        if (violation.frame is None or violation.signal is None or violation.signal.state != "RED"
                or violation.signal.timestamp != violation.frame_timestamp):
            LOGGER.warning("RED_LIGHT camera=%s track=%s result=REJECTED_INVALID_CROSSING_EVIDENCE",
                           camera.code, violation.track_id)
            return
        if len(self.pending) >= self.settings.max_pending:
            LOGGER.error("RED_LIGHT camera=%s track=%s result=DROPPED_PENDING_FULL", camera.code, violation.track.id)
            return
        key = f"{violation.track_key}:RED_LIGHT"
        if key in self.pending:
            return
        future = self.executor.submit(prepare_evidence, camera, violation, self.settings.snapshot_dir)
        # Future owns the frame until saved; do not retain a second reference for OCR wait.
        self.pending[key] = [camera, replace(violation, frame=None), time.monotonic(), future, None, 0., None, None]
        LOGGER.info("RED_LIGHT camera=%s track=%s result=PENDING_OCR_EVIDENCE", camera.code, violation.track.id)

    def cancel(self, camera_code):
        for key, item in tuple(self.pending.items()):
            if item[0].code == camera_code:
                item[3].cancel()
                del self.pending[key]
                LOGGER.warning("RED_LIGHT camera=%s object_key=%s result=CANCELLED_CONFIG", camera_code, key)

    def flush(self, now):
        for key, item in tuple(self.pending.items()):
            camera, violation, created, future, event, retry_at, manifest_future, upload_future = item
            age = now-created
            if age > self.settings.pending_ttl:
                LOGGER.error("RED_LIGHT camera=%s object_key=%s result=EXPIRED_NOT_DELIVERED", camera.code, key)
                if event is None and violation_read(violation, self.settings) is None:
                    LOGGER.warning("RED_LIGHT camera=%s object_key=%s result=REJECTED_UNVERIFIED_PLATE frame=%s",
                                   camera.code, key, violation.frame_sequence)
                future.cancel()
                del self.pending[key]
                continue
            if age < self.settings.ocr_wait_seconds or now < retry_at:
                continue
            try:
                if event is None:
                    if violation_read(violation, self.settings) is None:
                        continue
                    if not future.done():
                        continue
                    files = future.result()
                    assets = {}
                    if self.snapshots:
                        if item[7] is None:
                            # Lock the verified OCR before drawing/uploading the viewing copy.
                            violation = replace(violation, read_at_crossing=violation_read(violation, self.settings))
                            item[1] = violation
                            item[7] = self.executor.submit(upload_evidence, camera, files, self.snapshots, key,
                                                           violation, violation.read_at_crossing.read.license_plate.text)
                        if not item[7].done():
                            continue
                        ready, assets = self.snapshots.resolve(item[7].result(), age)
                        if not ready:
                            continue
                        assets = assets or {}
                        # Never present a marked copy as the only evidence asset.
                        snapshot = (assets.get("annotated") or assets.get("original")) if assets.get("original") else None
                    else:
                        snapshot = None
                    event = build_red_light_event(camera, violation, snapshot, self.settings)
                    if event is None:
                        continue
                    event["payload"]["violation_evidence"]["frame_sha256"] = files.frame_sha256
                    if assets.get("original"):
                        event["payload"]["violation_evidence"].update({
                            "original_snapshot_url": assets["original"],
                            "annotated_snapshot_url": assets.get("annotated"),
                            "snapshot_view": "ANNOTATED" if assets.get("annotated") else "ORIGINAL",
                        })
                    item[4] = event
                    item[6] = self.executor.submit(write_manifest, files, {
                        "camera_id": camera.id, "entity_id": event["entity_id"],
                        "event_time": event["event_time"], "snapshot_url": snapshot,
                        "license_plate": event["payload"]["vehicle"]["license_plate"],
                        "violation_evidence": event["payload"]["violation_evidence"]})
                if not item[6].done():
                    continue
                item[6].result()
                item[5] = now+self.settings.retry_seconds
                if self.publish(event, camera_code=camera.code, object_key=key):
                    del self.pending[key]
                    LOGGER.info("RED_LIGHT camera=%s object_key=%s result=QUEUED", camera.code, key)
            except Exception as error:
                LOGGER.error("RED_LIGHT camera=%s object_key=%s result=DELIVERY_FAILED type=%s",
                             camera.code, key, type(error).__name__)
                item[5] = now+self.settings.retry_seconds

    def stop(self):
        if self.pending:
            LOGGER.warning("RED_LIGHT result=STOPPED_NOT_PERSISTED pending=%d", len(self.pending))
        self.pending.clear()
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
