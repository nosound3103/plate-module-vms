"""Plate localization, character ordering and vehicle attribute inference."""
from dataclasses import dataclass, replace
import hashlib
import logging
import re
from plate_ai.detection.detector import VehicleDetection, YoloDetector
from plate_ai.core.domain import Zone

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlateRecognition:
    text: str = ""
    confidence: float = 0.
    bbox: tuple[int, int, int, int] | None = None
    detection_confidence: float = 0.
    plate_color: str | None = None

    @property
    def status(self):
        return "DETECTED" if self.text else ("UNREADABLE" if self.bbox else "NOT_DETECTED")


@dataclass(frozen=True)
class VehicleRecognition:
    detection_index: int
    vehicle: VehicleDetection
    license_plate: PlateRecognition
    manufacturer: str | None = None
    color: str | None = None
    object_key: str | None = None


def order_characters(detections, shape):
    """Suppress overlapping character hypotheses and order one or two rows."""
    import statistics
    unique = []
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        x1, y1, x2, y2 = detection.bbox
        duplicate = False
        for other in unique:
            ax1, ay1, ax2, ay2 = other.bbox
            overlap = max(0, min(x2, ax2)-max(x1, ax1))*max(0, min(y2, ay2)-max(y1, ay1))
            union = (x2-x1)*(y2-y1)+(ax2-ax1)*(ay2-ay1)-overlap
            if union > 0 and overlap/union > .5:
                duplicate = True
                break
        if not duplicate:
            unique.append(detection)
    ordered = sorted(unique, key=lambda d: d.bbox[0])
    if shape[1]/shape[0] <= 2 and len(unique) >= 5:
        by_y = sorted(unique, key=lambda d: d.bbox[1]+d.bbox[3])
        centers = [(d.bbox[1]+d.bbox[3])/2 for d in by_y]
        gaps = [b-a for a, b in zip(centers, centers[1:])]
        split = max(range(len(gaps)), key=gaps.__getitem__)+1
        if gaps[split-1] > .4*statistics.median(d.bbox[3]-d.bbox[1] for d in by_y):
            ordered = sorted(by_y[:split], key=lambda d: d.bbox[0])+sorted(by_y[split:], key=lambda d: d.bbox[0])
    return ordered


class AttributeClassifier:
    def __init__(self, detection_settings, artifact, confidence):
        from plate_ai.models import checkpoint
        import torch
        path = detection_settings.weights_dir/artifact.filename
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != artifact.sha256.lower():
                raise ValueError(f"{artifact.role} model checksum mismatch")
            handle.seek(0)
            state = torch.load(handle, map_location="cpu", pickle_module=checkpoint, weights_only=False)
        self.model = (state.get("ema") or state["model"]).float().to(detection_settings.device).eval()
        self.device, self.image_size = detection_settings.device, artifact.image_size
        self.confidence = confidence
        self.names = self.model.names
        LOGGER.info("Attribute model loaded role=%s weight=%s device=%s", artifact.role, artifact.filename, self.device)

    def classify(self, crop):
        import cv2
        import numpy as np
        import torch
        height, width = crop.shape[:2]
        size = min(height, width)
        if not size:
            return None
        top, left = (height-size)//2, (width-size)//2
        image = cv2.resize(crop[top:top+size, left:left+size], (self.image_size, self.image_size))
        array = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        with torch.inference_mode():
            tensor = torch.from_numpy(array).to(self.device).float()[None]/255.
            mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
            std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
            logits = self.model((tensor-mean)/std)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probability, index = logits.softmax(dim=1)[0].max(dim=0)
            if float(probability) < self.confidence:
                return None
            return str(self.names[int(index)]).strip().upper() or None


class RecognitionPipeline:
    def __init__(self, detection_settings, settings):
        self.settings = settings
        artifacts = {a.role: a for a in settings.artifacts}

        def detector(role, confidence):
            artifact = artifacts[role]
            return YoloDetector(replace(detection_settings, vehicle_weight=artifact.filename,
                                       vehicle_sha256=artifact.sha256, image_size=artifact.image_size,
                                       confidence=confidence))

        self.plate = detector("plate", settings.plate_confidence)
        self.characters = detector("digit", settings.character_confidence)
        self.brand = AttributeClassifier(detection_settings, artifacts["brand"], settings.attribute_confidence)
        self.color = AttributeClassifier(detection_settings, artifacts["color"], settings.attribute_confidence)
        self.plate_color = AttributeClassifier(detection_settings, artifacts["plate_color"], settings.attribute_confidence)

    def _read(self, crop):
        characters = order_characters(self.characters.detect(crop), crop.shape)
        if not characters:
            return "", 0.
        text = re.sub(r"[^A-Z0-9]", "", "".join(d.label.upper().strip() for d in characters))
        if not (6 <= len(text) <= 12 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text)):
            return "", 0.
        return text, sum(d.confidence for d in characters)/len(characters)

    def process(self, result, zones, *, detection_indices=None):
        import cv2
        import numpy as np
        frame = result.frame.image
        height, width = frame.shape[:2]
        polygons = [np.asarray([(x*(width-1), y*(height-1)) for x, y in zone.points], dtype=np.float32)
                    for zone in zones if isinstance(zone, Zone)]
        recognized = []
        selected = sorted(((i, v) for i, v in enumerate(result.detections)
                           if detection_indices is None or i in detection_indices),
                          key=lambda item: item[1].confidence, reverse=True)
        for index, vehicle in selected[:self.settings.max_vehicles]:
            x1, y1, x2, y2 = vehicle.bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            point = ((x1+x2)/2, float(min(y2, height-1)))
            if polygons and not any(cv2.pointPolygonTest(p, point, False) >= 0 for p in polygons):
                continue
            crop = frame[y1:y2, x1:x2]
            plate = PlateRecognition()
            candidates = sorted(self.plate.detect(crop), key=lambda p: p.confidence, reverse=True)
            for candidate in candidates[:self.settings.max_plates_per_vehicle]:
                px1, py1, px2, py2 = candidate.bbox
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(x2-x1, px2), min(y2-y1, py2)
                if px2-px1 < 25 or py2-py1 < 10:
                    continue
                plate_crop = crop[py1:py2, px1:px2]
                text, confidence = self._read(plate_crop)
                current = PlateRecognition(text, confidence, (x1+px1, y1+py1, x1+px2, y1+py2),
                                           candidate.confidence)
                if plate.bbox is None or text:
                    plate = current
                if text:
                    plate = replace(plate, plate_color=self.plate_color.classify(plate_crop))
                    break
            manufacturer, color = None, None
            if plate.text and vehicle.vehicle_type == "CAR":
                brand = self.brand.classify(crop)
                manufacturer = {"BWM": "BMW", "MADAZ": "MAZDA", "MESCEDES": "MERCEDES-BENZ",
                                "LAMORGHINI": "LAMBORGHINI"}.get(brand, brand)
                color = self.color.classify(crop)
            recognized.append(VehicleRecognition(index, vehicle, plate, manufacturer, color))
        return tuple(sorted(recognized, key=lambda r: r.detection_index))
