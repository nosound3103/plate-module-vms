"""Bounded Storage API upload policy for PLATE event snapshots."""
from dataclasses import dataclass, fields
import math
import os


@dataclass(frozen=True)
class SnapshotSettings:
    enabled: bool = True
    timeout_seconds: float = 4.
    attempts: int = 2
    retry_seconds: float = .5
    wait_seconds: float = 10.
    workers: int = 2
    max_pending: int = 64
    frame_buffer: int = 4
    jpeg_quality: int = 90
    max_bytes: int = 10*1024*1024

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "enabled":
                if type(value) is not bool:
                    raise ValueError("PLATE_SNAPSHOT_ENABLED must be boolean")
                continue
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Snapshot {field.name} must be positive and finite")
            if isinstance(field.default, int) and type(value) is not int:
                raise ValueError(f"Snapshot {field.name} must be integer")
        if self.jpeg_quality > 100 or self.max_bytes > 10*1024*1024:
            raise ValueError("Snapshot quality must be <=100 and size <=10 MB")

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        values = {}
        for field in fields(cls):
            name = f"PLATE_SNAPSHOT_{field.name.upper()}"
            raw = str(env.get(name, field.default)).strip()
            try:
                if field.name == "enabled":
                    if raw.lower() not in {"true", "false", "1", "0"}:
                        raise ValueError()
                    values[field.name] = raw.lower() in {"true", "1"}
                else:
                    values[field.name] = type(field.default)(raw)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid {name}") from None
        return cls(**values)
