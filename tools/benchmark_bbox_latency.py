"""Short live bbox/OCR benchmark. Publishes overlays, never events or snapshots."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import statistics
import time

from plate_ai.overlay.config import BboxSettings
from plate_ai.overlay.runtime import BboxRuntime
from plate_ai.core.config import Settings
from plate_ai.detection.config import DetectionSettings
from plate_ai.tracking.config import PassageSettings
from plate_ai.tracking.runtime import PassageRuntime
from plate_ai.recognition.config import RecognitionSettings
from plate_ai.recognition.runtime import RecognitionRuntime
from plate_ai.traffic_light.config import RedLightSettings
from plate_ai.detection.runtime import DetectionRuntime
from plate_ai.camera.sync import CameraSyncService


def stats(values):
    values = sorted(values)
    return {"n":len(values), "mean":round(statistics.mean(values), 2),
            "p50":round(statistics.median(values), 2),
            "p95":round(values[min(len(values)-1, int(len(values)*.95))], 2),
            "max":round(max(values), 2)} if values else {"n":0}


def main():
    logging.basicConfig(level=logging.WARNING)
    duration = float(os.environ.get("BBOX_BENCH_SECONDS", "45"))
    if not 0 < duration <= 300:
        raise ValueError("BBOX_BENCH_SECONDS must be in (0,300]")
    latencies, ages, inference, queued = [], [], [], set()
    with ExitStack() as stack:
        sync = stack.enter_context(CameraSyncService(Settings.from_env()))
        red = RedLightSettings.from_env()
        detection = stack.enter_context(DetectionRuntime(sync, DetectionSettings.from_env(), metadata_frames=64,
                                  metadata_seconds=4, evidence_frames=red.frame_buffer if red.enabled else 0))
        recognition = RecognitionRuntime(sync, detection, RecognitionSettings.from_env())
        passages = PassageRuntime(sync, detection, recognition, PassageSettings.from_env(), emit_events=False,
                                  publish=lambda *a, **k:True, red_light_settings=red)
        if passages.red_events:
            passages.red_events.submit = lambda *args:None
        stack.enter_context(passages)
        bbox = BboxRuntime(sync, passages, BboxSettings.from_env())
        publish = bbox.publish
        def measured_publish(message, **kwargs):
            snapshot = passages.latest_overlay(message["camera_code"])
            accepted = publish(message, **kwargs)
            now = time.monotonic()
            key = (message["camera_code"], message["frame_sequence"])
            if (accepted and snapshot is not None and snapshot.sequence == message["frame_sequence"]
                    and snapshot.detection_completed_monotonic and key not in queued):
                queued.add(key)
                latencies.append((now-snapshot.detection_completed_monotonic)*1000)
                ages.append(message["frame_age_ms"])
                result = detection.latest(message["camera_code"])
                if result is not None and result.frame.sequence == message["frame_sequence"]:
                    inference.append(result.inference_ms)
            return accepted
        bbox.publish = measured_publish
        stack.enter_context(bbox)
        recognition.track_source = passages
        stack.enter_context(recognition)
        print("BENCHMARK_STARTED: live detection/tracking/OCR/preview + bbox MQTT; events/snapshots disabled", flush=True)
        deadline = time.monotonic()+duration
        while time.monotonic() < deadline:
            time.sleep(.1)
        report = {"vehicle_weight":detection.settings.vehicle_weight, "device":detection.settings.device,
                  "image_size":detection.settings.image_size, "duration_seconds":duration,
                  "red_light_preview":red.enabled, "detection_to_mqtt_queue_ms":stats(latencies),
                  "frame_age_at_queue_ms":stats(ages), "vehicle_detection_ms":stats(inference)}
        output = Path("/snapshots/diagnostics")/("bbox_priority_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print("REPORT="+str(output), flush=True)


if __name__ == "__main__":
    main()
