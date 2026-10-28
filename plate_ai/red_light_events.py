"""PLATE-compatible evidence delivery using the existing asynchronous MQTT client."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import logging
import re
import time
from .events import build_plate_event
from .snapshots import save_original

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


def build_red_light_event(camera, violation, snapshot):
    track = violation.track
    read = track.best_read()
    if read is None or not re.sub(r"[^A-Z0-9]", "", read.license_plate.text.upper()):
        return None  # Unknown-plate backend violation contract is not present here.
    key = f"{track.object_key}:RED_LIGHT"
    view = replace(track, crossed_epoch=violation.timestamp, object_key=key, direction=track.direction or "IN")
    event = build_plate_event(camera, view)
    vehicle = event["payload"]["vehicle"]
    event["payload"]["vehicle"] = {k: v for k, v in vehicle.items() if v is not None}
    event["payload"].update({
        "violation_type_code": "RED_LIGHT",
        "violation_evidence": {
            "rule": "red_light_crossing", "confidence": round(violation.confidence, 4),
            "traffic_light_state": "RED", "from_zone": "STOP-ZONE", "to_zone": "VIOLATION-ZONE",
            "track_id": track.id, "crossing_timestamp": event["event_time"],
            "snapshot_timestamp": iso_timestamp(violation.frame_timestamp),
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
        self.executor = (None if snapshots else
                         ThreadPoolExecutor(max_workers=1, thread_name_prefix="red-light-evidence"))

    def submit(self, camera, violation):
        if len(self.pending) >= self.settings.max_pending:
            LOGGER.error("RED_LIGHT camera=%s track=%s result=DROPPED_PENDING_FULL", camera.code, violation.track.id)
            return
        key = f"{violation.track.object_key}:RED_LIGHT"
        if key in self.pending:
            return
        future = (self.snapshots.submit(camera, violation.frame, object_key=key, local_dir=self.settings.snapshot_dir)
                  if self.snapshots else self.executor.submit(save_original, violation.frame, self.settings.snapshot_dir))
        # Future owns the frame until saved; do not retain a second reference for OCR wait.
        self.pending[key] = [camera, replace(violation, frame=None), time.monotonic(), future, None, 0.]
        LOGGER.info("RED_LIGHT camera=%s track=%s result=PENDING_OCR_EVIDENCE", camera.code, violation.track.id)

    def cancel(self, camera_code):
        for key, item in tuple(self.pending.items()):
            if item[0].code == camera_code:
                item[3].cancel()
                del self.pending[key]
                LOGGER.warning("RED_LIGHT camera=%s object_key=%s result=CANCELLED_CONFIG", camera_code, key)

    def flush(self, now):
        for key, item in tuple(self.pending.items()):
            camera, violation, created, future, event, retry_at = item
            age = now-created
            if age > self.settings.pending_ttl:
                LOGGER.error("RED_LIGHT camera=%s object_key=%s result=EXPIRED_NOT_DELIVERED", camera.code, key)
                future.cancel()
                del self.pending[key]
                continue
            if age < self.settings.ocr_wait_seconds or now < retry_at:
                continue
            try:
                if event is None:
                    if violation.track.best_read() is None:
                        continue
                    if self.snapshots:
                        ready, snapshot = self.snapshots.resolve(future, age)
                        if not ready:
                            continue
                    else:
                        if not future.done():
                            continue
                        future.result()  # Keep local evidence, never publish a filesystem path.
                        snapshot = None
                    event = build_red_light_event(camera, violation, snapshot)
                    if event is None:
                        continue
                    item[4] = event
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
