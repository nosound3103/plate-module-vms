"""Configuration for RTSP acquisition and vehicle inference."""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class DetectionSettings:
    device: str = "cpu"
    weights_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1]/"models"/"weights")
    vehicle_weight: str = "last_vehicle_s_07042023_640_hp.pt"
    vehicle_sha256: str = "1dd3770b89f3517c81e1d1101dd7815b0c0fc4b16741b1769a655168c95673c7"
    image_size: int = 640
    confidence: float = .4
    iou: float = .45
    max_detections: int = 256
    max_fps: float = 5.
    max_frame_age: float = 1.
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 2000
    rtsp_transport: str = "tcp"
    torch_threads: int = 1
    shutdown_timeout_seconds: float = 12.
    vehicle_engine: str = ""
    vehicle_engine_sha256: str = ""

    def __post_init__(self):
        if self.vehicle_engine:
            if Path(self.vehicle_engine).name != self.vehicle_engine or not re.fullmatch(r"[a-fA-F0-9]{64}", self.vehicle_engine_sha256):
                raise ValueError("Vehicle engine requires a filename and SHA-256 digest")
            if self.device == "cpu":
                raise ValueError("TensorRT vehicle engine requires CUDA")
        if self.device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", self.device):
            raise ValueError("PLATE_DEVICE must be cpu, cuda or cuda:<index>")
        if not self.vehicle_weight or Path(self.vehicle_weight).name != self.vehicle_weight:
            raise ValueError("PLATE_VEHICLE_WEIGHT must be a filename")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.vehicle_sha256):
            raise ValueError("PLATE_VEHICLE_SHA256 must be a SHA-256 hex digest")
        for name in ("max_fps", "max_frame_age", "shutdown_timeout_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.confidence <= 1 or not 0 < self.iou <= 1:
            raise ValueError("Detection confidence and IoU must be in (0, 1]")
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("PLATE_IMAGE_SIZE must be a positive multiple of 32")
        if min(self.max_detections, self.open_timeout_ms, self.read_timeout_ms, self.torch_threads) <= 0:
            raise ValueError("Detection limits and RTSP timeouts must be positive")
        if self.rtsp_transport not in {"tcp", "udp"}:
            raise ValueError("PLATE_RTSP_TRANSPORT must be tcp or udp")

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ

        def read(name, default):
            return env.get(name, "").strip() or str(default)

        def number(name, convert, default):
            try:
                return convert(read(name, default))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric environment variable: {name}") from None

        return cls(
            device=read("PLATE_DEVICE", cls.device),
            weights_dir=Path(read("PLATE_WEIGHTS_DIR", Path(__file__).resolve().parents[1]/"models"/"weights")),
            vehicle_weight=read("PLATE_VEHICLE_WEIGHT", cls.vehicle_weight),
            vehicle_sha256=read("PLATE_VEHICLE_SHA256", cls.vehicle_sha256),
            image_size=number("PLATE_IMAGE_SIZE", int, cls.image_size),
            confidence=number("PLATE_VEHICLE_CONFIDENCE", float, cls.confidence),
            iou=number("PLATE_VEHICLE_IOU", float, cls.iou),
            max_detections=number("PLATE_MAX_DETECTIONS", int, cls.max_detections),
            max_fps=number("PLATE_MAX_FPS", float, cls.max_fps),
            max_frame_age=number("PLATE_MAX_FRAME_AGE_SECONDS", float, cls.max_frame_age),
            open_timeout_ms=number("PLATE_RTSP_OPEN_TIMEOUT_MS", int, cls.open_timeout_ms),
            read_timeout_ms=number("PLATE_RTSP_READ_TIMEOUT_MS", int, cls.read_timeout_ms),
            rtsp_transport=read("PLATE_RTSP_TRANSPORT", cls.rtsp_transport),
            torch_threads=number("PLATE_TORCH_THREADS", int, cls.torch_threads),
            shutdown_timeout_seconds=number("PLATE_SHUTDOWN_TIMEOUT_SECONDS", float, cls.shutdown_timeout_seconds),
            vehicle_engine=env.get("PLATE_VEHICLE_ENGINE", "").strip(),
            vehicle_engine_sha256=env.get("PLATE_VEHICLE_ENGINE_SHA256", "").strip(),
        )
