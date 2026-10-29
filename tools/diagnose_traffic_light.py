"""Read-only live HSV diagnostics; no AI inference or event publishing."""
from dataclasses import replace
from pathlib import Path
import json
import os
import time
import cv2
import numpy as np
from plate_ai.core.config import Settings
from plate_ai.camera.sync import CameraSyncService
from plate_ai.core.domain import TrafficZone
from plate_ai.detection.config import DetectionSettings
from plate_ai.camera.stream import LatestFrameStream
from plate_ai.traffic_light.detector import TrafficLightStateDetector, contour
from plate_ai.traffic_light.config import RedLightSettings


def main():
    settings = replace(Settings.from_env(), client_id="tts_light_diagnostic_01")
    output = Path("/snapshots/diagnostics")/str(int(time.time()))
    output.mkdir(parents=True, exist_ok=True)
    detector = TrafficLightStateDetector(RedLightSettings.from_env())
    records = []
    sample_count = min(300, max(1, int(os.environ.get("DIAGNOSTIC_SAMPLES", "50"))))
    with CameraSyncService(settings) as sync:
        deadline = time.monotonic()+20
        config = light = None
        while time.monotonic() < deadline:
            for candidate in sync.snapshot().cameras:
                lights = [z for z in candidate.zones if isinstance(z, TrafficZone) and z.name == "TRAFFICLIGHT" and z.points]
                if len(lights) == 1:
                    config, light = candidate, lights[0]
                    break
            if light:
                break
            time.sleep(.1)
        if light is None:
            raise RuntimeError("No valid configured traffic light polygon")
        stream = LatestFrameStream(config.camera, DetectionSettings.from_env())
        stream.start()
        try:
            deadline = time.monotonic()+sample_count*.25+8
            while len(records) < sample_count and time.monotonic() < deadline:
                frame = stream.latest()
                if frame is None:
                    time.sleep(.02)
                    continue
                image = frame.image
                h, w = image.shape[:2]
                polygon = np.rint(contour(light.points)*[w-1, h-1]).astype(np.int32)
                x, y, bw, bh = cv2.boundingRect(polygon)
                crop = image[y:y+bh, x:x+bw]
                mask = np.zeros((bh, bw), np.uint8)
                cv2.fillPoly(mask, [polygon-[x, y]], 255)
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                selected = hsv[(mask > 0) & (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 50)]
                hist = np.bincount(selected[:, 0], minlength=180) if len(selected) else np.zeros(180)
                sample = detector.detect(image, light.points, frame.captured_at)
                components = {}
                for name, ranges in (("RED", detector.settings.red_hues), ("GREEN", detector.settings.green_hues), ("YELLOW", detector.settings.yellow_hues)):
                    color = np.zeros_like(mask)
                    for low, high in ranges:
                        color |= cv2.inRange(hsv, (low, detector.settings.saturation_min, detector.settings.value_min), (high, 255, 255))
                    color &= mask
                    n, labels, stats, _ = cv2.connectedComponentsWithStats(color, connectivity=8)
                    items = []
                    for i in range(1, n):
                        cx, cy, cw, ch, area = map(int, stats[i])
                        if area >= 2:
                            pixels = hsv[labels == i]
                            items.append({"area": area, "ratio": area/int(np.count_nonzero(mask)),
                                          "fill": area/(cw*ch), "bbox": [cx, cy, cw, ch],
                                          "hsv_mean": pixels.mean(axis=0).round(1).tolist()})
                    components[name] = sorted(items, key=lambda item: -item["area"])[:5]
                record = {"timestamp": frame.captured_at, "frame_shape": [h, w], "roi": [x, y, bw, bh],
                          "state": sample.state, "confidence": sample.confidence, "scores": sample.scores,
                          "top_hues": [(int(i), int(hist[i])) for i in np.argsort(hist)[-12:][::-1]],
                          "components": components}
                index = len(records)
                if index == 0:
                    cv2.imwrite(str(output/"original.png"), image)
                cv2.imwrite(str(output/f"roi-{index:03d}.png"), crop)
                records.append(record)
                time.sleep(.2)
        finally:
            stream.stop(5)
    (output/"distribution.json").write_text(json.dumps({"camera_id": config.camera.id,
                                                      "points": light.points, "samples": records}, indent=2))
    print(json.dumps({"output": str(output), "samples": len(records), "first": records[0] if records else None}, indent=2))


if __name__ == "__main__":
    main()
