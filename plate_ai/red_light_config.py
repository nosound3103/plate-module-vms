"""HSV and conservative crossing policy; no extra inference model."""
from dataclasses import dataclass, fields
import json
import math
import os


@dataclass(frozen=True)
class RedLightSettings:
    enabled: bool = False
    saturation_min: int = 100
    value_min: int = 150
    red_hues: tuple = ((0, 7), (170, 179))
    yellow_hues: tuple = ((8, 35),)
    green_hues: tuple = ((36, 90),)
    min_roi_pixels: int = 64
    min_component_pixels: int = 6
    min_component_ratio: float = .005
    min_component_fill: float = .25
    component_close_size: int = 5
    min_component_support: float = .35
    min_confidence: float = .45
    ambiguity_ratio: float = .65
    signal_confirm_frames: int = 2
    signal_max_age: float = .5
    zone_confirm_frames: int = 2
    boundary_margin: float = .005
    max_zone_gap: float = .005
    max_crossing_seconds: float = .5
    frame_buffer: int = 4
    ocr_wait_seconds: float = 2.
    pending_ttl: float = 30.
    max_pending: int = 64
    retry_seconds: float = 1.
    dedup_guard_seconds: float = 1.25
    snapshot_dir: str = "snapshots/red_light"
    debug: bool = False

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"RED_LIGHT {f.name} must be positive and finite")
        for name in ("enabled", "debug"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"RED_LIGHT {name} must be boolean")
        for name in ("signal_confirm_frames", "zone_confirm_frames", "frame_buffer", "max_pending",
                     "min_roi_pixels", "min_component_pixels", "saturation_min", "value_min", "component_close_size"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"RED_LIGHT {name} must be integer")
        if self.signal_confirm_frames < 2 or self.zone_confirm_frames < 2:
            raise ValueError("RED_LIGHT confirmation requires at least two frames")
        if max(self.saturation_min, self.value_min) > 255:
            raise ValueError("HSV S/V must be <=255")
        if self.component_close_size % 2 != 1 or self.component_close_size > 15:
            raise ValueError("RED_LIGHT component_close_size must be odd and <=15 (1 disables closing)")
        for name in ("min_component_ratio", "min_component_fill", "min_component_support", "min_confidence", "ambiguity_ratio",
                     "boundary_margin", "max_zone_gap"):
            if getattr(self, name) >= 1:
                raise ValueError(f"RED_LIGHT {name} must be <1")
        for name in ("red_hues", "yellow_hues", "green_hues"):
            ranges = getattr(self, name)
            if not ranges or any(len(r) != 2 or not 0 <= r[0] <= r[1] <= 179 for r in ranges):
                raise ValueError(f"RED_LIGHT invalid {name}")
        if self.pending_ttl <= self.ocr_wait_seconds or self.dedup_guard_seconds <= 1:
            raise ValueError("RED_LIGHT TTL must exceed OCR wait; dedup guard must exceed 1s")
        if not isinstance(self.snapshot_dir, str) or not self.snapshot_dir.strip():
            raise ValueError("RED_LIGHT snapshot_dir required")

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        enabled = env.get("RED_LIGHT_ENABLED", "false").lower().strip()
        if enabled not in {"true", "false", "1", "0"}:
            raise ValueError("RED_LIGHT_ENABLED must be boolean")
        try:
            values = json.loads(env.get("RED_LIGHT_SETTINGS", "{}"))
            return cls(**{**values, "enabled": enabled in {"true", "1"}})
        except (TypeError, ValueError, KeyError):
            raise ValueError("Invalid RED_LIGHT_SETTINGS JSON configuration") from None
