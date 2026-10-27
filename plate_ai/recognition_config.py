"""Configuration and artifact identities for plate recognition."""
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class ModelArtifact:
    role: str
    filename: str
    sha256: str
    image_size: int

    def __post_init__(self):
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError(f"Invalid {self.role} model filename")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.sha256):
            raise ValueError(f"Invalid {self.role} model checksum")


ARTIFACTS = (
    ModelArtifact("plate", "last_plate_n_12012023.pt", "088c8b789d984b3723be242899015c654373e886cb15801c70c16ff2c5b4a01f", 256),
    ModelArtifact("digit", "last_digit_n_12012022.pt", "acc35369a568e03e062b94dbb28ce5b72be3ab029aea48fdfaa0371f854b56e4", 256),
    ModelArtifact("brand", "brand_v2_320.pt", "0ba727f6a095039c1974e2df3a3034ca9bef41f2fb7ee273183e9ef5929a6ff4", 320),
    ModelArtifact("color", "color_v2_224.pt", "4319bede403672fedef98a9ae013a1d80e2d4751bb5472a36f1d2735d531892a", 224),
    ModelArtifact("plate_color", "last_color_29122022_v2.pt", "b7f882fb023aa75419752fb6e3038f5c0d9a5dfa63d49c1400ddf9d49f0f7a58", 128),
)


@dataclass(frozen=True)
class RecognitionSettings:
    enabled: bool = False
    artifacts: tuple[ModelArtifact, ...] = ARTIFACTS
    plate_confidence: float = .55
    character_confidence: float = .35
    attribute_confidence: float = .5
    interval_seconds: float = .5
    max_result_age: float = 2.
    max_vehicles: int = 16
    max_plates_per_vehicle: int = 3

    def __post_init__(self):
        if {a.role for a in self.artifacts} != {a.role for a in ARTIFACTS} or len(self.artifacts) != len(ARTIFACTS):
            raise ValueError("Recognition requires five unique model roles")
        for name in ("plate_confidence", "character_confidence", "attribute_confidence"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("interval_seconds", "max_result_age"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if min(self.max_vehicles, self.max_plates_per_vehicle) <= 0:
            raise ValueError("Recognition limits must be positive")

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ

        def read(key, default):
            return env.get(key, "").strip() or str(default)

        enabled = read("PLATE_OCR_ENABLED", "false").lower()
        if enabled not in {"true", "false", "1", "0"}:
            raise ValueError("PLATE_OCR_ENABLED must be true or false")

        def number(key, convert, default):
            try:
                return convert(read(key, default))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric environment variable: {key}") from None

        return cls(
            enabled=enabled in {"true", "1"},
            artifacts=tuple(replace(a, filename=read(f"PLATE_{a.role.upper()}_WEIGHT", a.filename),
                                    sha256=read(f"PLATE_{a.role.upper()}_SHA256", a.sha256)) for a in ARTIFACTS),
            plate_confidence=number("PLATE_PLATE_CONFIDENCE", float, cls.plate_confidence),
            character_confidence=number("PLATE_CHARACTER_CONFIDENCE", float, cls.character_confidence),
            attribute_confidence=number("PLATE_ATTRIBUTE_CONFIDENCE", float, cls.attribute_confidence),
            interval_seconds=number("PLATE_OCR_INTERVAL_SECONDS", float, cls.interval_seconds),
            max_result_age=number("PLATE_OCR_MAX_RESULT_AGE_SECONDS", float, cls.max_result_age),
            max_vehicles=number("PLATE_OCR_MAX_VEHICLES", int, cls.max_vehicles),
            max_plates_per_vehicle=number("PLATE_OCR_MAX_PLATES_PER_VEHICLE", int, cls.max_plates_per_vehicle),
        )
