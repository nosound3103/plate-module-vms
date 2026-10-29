"""Vehicle inference with local checkpoint verification and class-aware NMS."""
from dataclasses import dataclass
import hashlib
import logging

LOGGER = logging.getLogger(__name__)
VEHICLE_TYPES = {"car": "CAR", "motorbike": "MOTORBIKE", "motorcycle": "MOTORBIKE",
                 "bus": "BUS", "truck": "TRUCK"}


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class VehicleDetection:
    vehicle_type: str
    confidence: float
    bbox: tuple[int, int, int, int]


class YoloDetector:
    def __init__(self, settings, *, class_labels=None):
        from plate_ai.models import checkpoint
        import torch
        self.settings = settings
        self.torch = torch
        self.device = torch.device(settings.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; enable GPU access or select PLATE_DEVICE=cpu")
        path = settings.weights_dir/settings.vehicle_weight
        if not path.is_file():
            raise FileNotFoundError(f"Detection weight missing: {settings.vehicle_weight}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != settings.vehicle_sha256.lower():
                raise ValueError("Detection weight checksum mismatch")
            handle.seek(0)
            state = torch.load(handle, map_location="cpu", pickle_module=checkpoint, weights_only=False)
        self.model = (state.get("ema") or state["model"]).float().to(self.device).eval()
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Upsample) and not hasattr(layer, "recompute_scale_factor"):
                layer.recompute_scale_factor = None
        self.names = self.model.names
        names = self.names.items() if isinstance(self.names, dict) else enumerate(self.names)
        self.classes = {int(k): (class_labels[str(v).lower().strip()] if class_labels is not None else str(v))
                        for k, v in names if class_labels is None or str(v).lower().strip() in class_labels}
        if not self.classes:
            raise ValueError("Detection model has no supported classes")
        self.stride = max(32, int(self.model.stride.max()))
        self.image_size = ((settings.image_size+self.stride-1)//self.stride)*self.stride
        LOGGER.info("Detection model loaded weight=%s device=%s classes=%s",
                    settings.vehicle_weight, settings.device, ",".join(self.classes.values()))

    def detect(self, image):
        import cv2
        import numpy as np
        from torchvision.ops import batched_nms
        torch = self.torch
        height, width = image.shape[:2]
        scale = min(self.image_size/width, self.image_size/height)
        resized_width, resized_height = round(width*scale), round(height*scale)
        padding = (self.image_size-resized_width, self.image_size-resized_height)
        if not getattr(self, "square_input", False):
            padding = tuple(value % self.stride for value in padding)
        dw, dh = padding[0]/2, padding[1]/2
        resized = cv2.resize(image, (resized_width, resized_height))
        padded = cv2.copyMakeBorder(resized, round(dh-.1), round(dh+.1), round(dw-.1), round(dw+.1),
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
        with torch.inference_mode():
            tensor = torch.from_numpy(array).to(self.device).float()[None]/255.
            prediction = self.model(tensor)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            prediction = prediction[0]
            prediction = prediction[prediction[:, 4] > self.settings.confidence]
            if not len(prediction):
                return ()
            scores, classes = (prediction[:, 5:]*prediction[:, 4:5]).max(dim=1)
            valid = scores > self.settings.confidence
            supported = torch.zeros_like(valid)
            for identity in self.classes:
                supported |= classes == identity
            valid &= supported
            prediction, scores, classes = prediction[valid], scores[valid], classes[valid]
            if not len(prediction):
                return ()
            top = scores.argsort(descending=True)[:30000]
            center, size = prediction[top, :2], prediction[top, 2:4]
            boxes = torch.cat((center-size/2, center+size/2), dim=1)
            scores, classes = scores[top], classes[top]
            keep = batched_nms(boxes, scores, classes, self.settings.iou)[:self.settings.max_detections]
            boxes = boxes[keep]
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]]-dw)/scale).clamp(0, width)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]]-dh)/scale).clamp(0, height)
            result = []
            for box, score, identity in zip(boxes.round().cpu().tolist(), scores[keep].cpu().tolist(), classes[keep].cpu().tolist()):
                if box[2] > box[0] and box[3] > box[1]:
                    result.append(ObjectDetection(self.classes[identity], float(score), tuple(int(v) for v in box)))
            return tuple(result)


class VehicleDetector(YoloDetector):
    def __init__(self, settings):
        super().__init__(settings, class_labels=VEHICLE_TYPES)
        if settings.vehicle_engine:
            from plate_ai.detection.tensorrt import TensorRTEngine
            self.model = TensorRTEngine(settings.weights_dir/settings.vehicle_engine,
                                        settings.vehicle_engine_sha256, self.device, self.image_size,
                                        max(self.classes)+1)
            self.square_input = True
            self.torch.cuda.empty_cache()
            LOGGER.info("Vehicle backend=TENSORRT engine=%s input=%sx%s", settings.vehicle_engine,
                        self.image_size, self.image_size)

    def detect(self, image):
        return tuple(VehicleDetection(d.label, d.confidence, d.bbox) for d in super().detect(image))
