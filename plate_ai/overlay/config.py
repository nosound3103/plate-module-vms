"""Realtime overlay cadence, freshness and MQTT routing."""
from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class BboxSettings:
    enabled: bool = False
    max_fps: float = 10.
    max_frame_age: float = .3
    plate_max_age: float = 1.
    topic_prefix: str = "smart_vms/ai/bbox"
    burst_capacity: int = 2

    def __post_init__(self):
        for name in ("max_fps", "max_frame_age", "plate_max_age"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.topic_prefix not in {"smart_vms/ai/bbox", "smart_vms/bbox"}:
            raise ValueError("PLATE_BBOX_TOPIC_PREFIX must be smart_vms/ai/bbox or smart_vms/bbox")
        if type(self.burst_capacity) is not int or self.burst_capacity < 2:
            raise ValueError("PLATE_BBOX_BURST_CAPACITY must be an integer >=2")

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ

        def read(key, default):
            return env.get(key, "").strip() or str(default)

        enabled = read("PLATE_BBOX_ENABLED", "false").lower()
        if enabled not in {"true", "false", "1", "0"}:
            raise ValueError("PLATE_BBOX_ENABLED must be true or false")

        def number(key, default):
            try:
                return float(read(key, default))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric environment variable: {key}") from None

        try:
            burst = int(read("PLATE_BBOX_BURST_CAPACITY", cls.burst_capacity))
        except ValueError:
            raise ValueError("Invalid PLATE_BBOX_BURST_CAPACITY") from None
        return cls(enabled=enabled in {"true", "1"},
                   max_fps=number("PLATE_BBOX_MAX_FPS", cls.max_fps),
                   max_frame_age=number("PLATE_BBOX_MAX_FRAME_AGE_SECONDS", cls.max_frame_age),
                   plate_max_age=number("PLATE_BBOX_PLATE_MAX_AGE_SECONDS", cls.plate_max_age),
                   topic_prefix=read("PLATE_BBOX_TOPIC_PREFIX", cls.topic_prefix), burst_capacity=burst)
