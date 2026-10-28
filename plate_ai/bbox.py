"""Immutable tracked geometry and normalized VMS overlay serialization."""
from dataclasses import dataclass
import time
from .domain import Camera


@dataclass(frozen=True)
class OverlayDetection:
    object_key: str
    track_id: int
    vehicle_type: str
    confidence: float
    bbox: tuple[float, float, float, float]
    plate_relative: tuple | None
    plate_seen_at: float
    plate_text: str
    plate_confidence: float
    text_confidence: float


@dataclass(frozen=True)
class OverlaySnapshot:
    camera: Camera
    sequence: int
    captured_at: float
    received_monotonic: float
    detections: tuple[OverlayDetection, ...]
    count_in: int
    count_out: int
    traffic_light_points: tuple = ()
    traffic_light: object = None


def normalized_box(box):
    values = [round(max(0., min(1., v)), 6) for v in box]
    return values if values[2] > values[0] and values[3] > values[1] else None


def build_bbox_message(snapshot, organization_id, settings, *, clear=False):
    now = time.monotonic()
    detections = []
    if not clear:
        if snapshot.traffic_light_points and snapshot.traffic_light is not None:
            from .red_light import COLORS, LABELS
            sample = snapshot.traffic_light
            xs, ys = zip(*snapshot.traffic_light_points)
            b, g, r = COLORS[sample.state]
            key = f"{snapshot.camera.id}:TRAFFICLIGHT"
            detections.append({"id": key, "global_id": key,
                               "class": LABELS[sample.state], "text": LABELS[sample.state],
                               "traffic_light_state": sample.state,
                               "confidence": sample.confidence,
                               "bbox": normalized_box((min(xs), min(ys), max(xs), max(ys))),
                               "color": f"#{r:02x}{g:02x}{b:02x}"})
        for item in snapshot.detections:
            box = normalized_box(item.bbox)
            if box is None:
                continue
            detections.append({"id": item.object_key, "global_id": item.object_key,
                               "track_id": item.track_id, "class": item.vehicle_type.lower(),
                               "confidence": round(item.confidence, 4), "bbox": box})
            if item.plate_relative is None or now-item.plate_seen_at > settings.plate_max_age:
                continue
            x1, y1, x2, y2 = item.bbox
            rx1, ry1, rx2, ry2 = item.plate_relative
            plate_box = normalized_box((x1+rx1*(x2-x1), y1+ry1*(y2-y1),
                                        x1+rx2*(x2-x1), y1+ry2*(y2-y1)))
            if plate_box is not None:
                key = f"{item.object_key}:plate"
                detections.append({"id": key, "global_id": key, "track_id": item.track_id,
                                   "class": f"license_plate: {item.plate_text}" if item.plate_text else "license_plate",
                                   "text": item.plate_text, "confidence": round(item.plate_confidence, 4),
                                   "text_confidence": round(item.text_confidence, 4), "bbox": plate_box})
    return {
        "camera_code": snapshot.camera.code, "camera_id": snapshot.camera.id,
        "organization_id": organization_id, "ai_modules": "PLATE",
        "timestamp": time.time() if clear else snapshot.captured_at,
        "published_at": time.time(), "frame_sequence": snapshot.sequence,
        "frame_age_ms": round(max(0., now-snapshot.received_monotonic)*1000, 1),
        "detections": detections,
        "hud": {"text": f"IN: {snapshot.count_in} | OUT: {snapshot.count_out}",
                "display_ms": 350, "color": "#22c55e"},
    }
