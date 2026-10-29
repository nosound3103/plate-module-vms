"""Signal-aware CLI for configuration synchronization and vehicle detection."""
import argparse
from contextlib import ExitStack
import logging
import signal
import threading
import time
from plate_ai.core.config import Settings
from plate_ai.camera.sync import CameraSyncService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sync", "detection"), default="sync")
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        detection_settings = None
        recognition_settings = None
        passage_settings = None
        bbox_settings = None
        red_light_settings = None
        snapshot_settings = None
        delivery_settings = None
        tracking_enabled = False
        if args.mode == "detection":
            from plate_ai.detection.config import DetectionSettings
            from plate_ai.recognition.config import RecognitionSettings
            from plate_ai.tracking.config import PassageSettings
            from plate_ai.overlay.config import BboxSettings
            from plate_ai.traffic_light.config import RedLightSettings
            from plate_ai.evidence.config import SnapshotSettings
            from plate_ai.messaging.delivery import EventDeliverySettings
            detection_settings = DetectionSettings.from_env()
            recognition_settings = RecognitionSettings.from_env()
            passage_settings = PassageSettings.from_env()
            bbox_settings = BboxSettings.from_env()
            red_light_settings = RedLightSettings.from_env()
            snapshot_settings = SnapshotSettings.from_env()
            delivery_settings = EventDeliverySettings.from_env()
            if (snapshot_settings.enabled or red_light_settings.upload_snapshots) and (
                    (passage_settings.enabled and snapshot_settings.wait_seconds >= passage_settings.pending_ttl)
                    or (red_light_settings.enabled and snapshot_settings.wait_seconds >= red_light_settings.pending_ttl)):
                raise ValueError("PLATE_SNAPSHOT_WAIT_SECONDS must be less than the event pending TTL")
            tracking_enabled = (recognition_settings.enabled or passage_settings.enabled
                                or bbox_settings.enabled or red_light_settings.enabled)
            if red_light_settings.enabled and not recognition_settings.enabled:
                raise ValueError("RED_LIGHT_ENABLED requires PLATE_OCR_ENABLED=true")
            if passage_settings.enabled and not recognition_settings.enabled:
                raise ValueError("PLATE_EVENTS_ENABLED requires PLATE_OCR_ENABLED=true")
            if tracking_enabled and passage_settings.history_seconds < recognition_settings.max_result_age:
                raise ValueError("PLATE_TRACK_HISTORY_SECONDS must cover PLATE_OCR_MAX_RESULT_AGE_SECONDS")
    except ValueError as error:
        # Settings errors name the field, never the supplied value.
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error("Configuration error: %s", error)
        return 2
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)sZ %(levelname)s %(name)s %(message)s")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        with ExitStack() as stack:
            sync = stack.enter_context(CameraSyncService(settings))
            if detection_settings is not None:
                from plate_ai.detection.runtime import DetectionRuntime
                detection = stack.enter_context(DetectionRuntime(
                    sync, detection_settings,
                    metadata_frames=passage_settings.history_frames if tracking_enabled else 0,
                    metadata_seconds=passage_settings.history_seconds,
                    evidence_frames=max(red_light_settings.frame_buffer if red_light_settings.enabled else 0,
                                        snapshot_settings.frame_buffer if snapshot_settings.enabled and passage_settings.enabled else 0)))
                recognition = None
                if recognition_settings.enabled:
                    from plate_ai.recognition.runtime import RecognitionRuntime
                    recognition = RecognitionRuntime(sync, detection, recognition_settings)
                if tracking_enabled:
                    from plate_ai.tracking.runtime import PassageRuntime
                    snapshots = None
                    red_snapshots = None
                    if ((snapshot_settings.enabled and (passage_settings.enabled or red_light_settings.enabled))
                            or (red_light_settings.enabled and red_light_settings.upload_snapshots)):
                        from dataclasses import replace
                        from plate_ai.evidence.storage import SnapshotService
                        storage = stack.enter_context(SnapshotService(settings.broker_host,
                                                                      replace(snapshot_settings, enabled=True)))
                        snapshots = storage if snapshot_settings.enabled else None
                        red_snapshots = storage if red_light_settings.enabled else None
                    publish = None
                    if passage_settings.enabled or red_light_settings.enabled:
                        from plate_ai.messaging.delivery import EventDelivery
                        publish = stack.enter_context(EventDelivery(
                            delivery_settings, sync.transport.publish_plate_event,
                            sync.transport.publication_status,
                            eligible=lambda identity, code: any(c.camera.id == identity and c.camera.code == code
                                                                 for c in sync.snapshot().cameras),
                            guard_seconds=red_light_settings.dedup_guard_seconds,
                            recognition_enabled=passage_settings.enabled))
                        if red_light_settings.enabled:
                            logging.getLogger(__name__).warning(
                                "RED_LIGHT requires backend event-kind dedup and entity_id idempotency; "
                                "broker PUBACK does not confirm violation case creation")
                    passages = stack.enter_context(PassageRuntime(sync, detection, recognition, passage_settings,
                                                                 emit_events=passage_settings.enabled,
                                                                 red_light_settings=red_light_settings,
                                                                 snapshots=snapshots, red_light_snapshots=red_snapshots,
                                                                 publish=publish))
                    if bbox_settings.enabled:
                        from plate_ai.overlay.runtime import BboxRuntime
                        stack.enter_context(BboxRuntime(sync, passages, bbox_settings))
                    if recognition is not None:
                        recognition.track_source = passages
                        # Tracking/bbox are live before loading or running OCR models.
                        stack.enter_context(recognition)
            while not stop.wait(.5):
                pass
    except Exception as error:
        logging.getLogger(__name__).error("Service failed mode=%s type=%s", args.mode, type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
