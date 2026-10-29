"""Bounded tracking and PLATE event delivery configuration."""
from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class PassageSettings:
    enabled: bool = False
    track_ttl: float = 2.
    max_tracks: int = 128
    crossing_deadband: float = .01
    ocr_wait_seconds: float = 2.
    pending_ttl: float = 30.
    max_pending: int = 128
    retry_seconds: float = 1.
    history_seconds: float = 4.
    history_frames: int = 64
    identity_cost_margin: float = .1
    identity_max_gap_seconds: float = .5

    def __post_init__(self):
        for name in ("track_ttl", "ocr_wait_seconds", "pending_ttl", "retry_seconds", "history_seconds",
                     "identity_cost_margin", "identity_max_gap_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.crossing_deadband < .5:
            raise ValueError("PLATE_CROSSING_DEADBAND must be in [0, 0.5)")
        if min(self.max_tracks, self.max_pending, self.history_frames) <= 0:
            raise ValueError("Passage limits must be positive")
        if self.pending_ttl <= self.ocr_wait_seconds:
            raise ValueError("PLATE_EVENT_PENDING_TTL_SECONDS must exceed OCR wait")

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ

        def read(key, default):
            return env.get(key, "").strip() or str(default)

        enabled = read("PLATE_EVENTS_ENABLED", "false").lower()
        if enabled not in {"true", "false", "1", "0"}:
            raise ValueError("PLATE_EVENTS_ENABLED must be true or false")

        def number(key, convert, default):
            try:
                return convert(read(key, default))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric environment variable: {key}") from None

        return cls(
            enabled=enabled in {"true", "1"},
            track_ttl=number("PLATE_TRACK_TTL_SECONDS", float, cls.track_ttl),
            max_tracks=number("PLATE_MAX_TRACKS", int, cls.max_tracks),
            crossing_deadband=number("PLATE_CROSSING_DEADBAND", float, cls.crossing_deadband),
            ocr_wait_seconds=number("PLATE_EVENT_OCR_WAIT_SECONDS", float, cls.ocr_wait_seconds),
            pending_ttl=number("PLATE_EVENT_PENDING_TTL_SECONDS", float, cls.pending_ttl),
            max_pending=number("PLATE_EVENT_MAX_PENDING", int, cls.max_pending),
            retry_seconds=number("PLATE_EVENT_RETRY_SECONDS", float, cls.retry_seconds),
            history_seconds=number("PLATE_TRACK_HISTORY_SECONDS", float, cls.history_seconds),
            history_frames=number("PLATE_TRACK_HISTORY_FRAMES", int, cls.history_frames),
            identity_cost_margin=number("PLATE_IDENTITY_COST_MARGIN", float, cls.identity_cost_margin),
            identity_max_gap_seconds=number("PLATE_IDENTITY_MAX_GAP_SECONDS", float, cls.identity_max_gap_seconds),
        )
