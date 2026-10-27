"""PLATE integration foundation; importing it opens no network connections."""
from .config import Settings
from .service import CameraSyncService, CameraConfiguration
from .detection_config import DetectionSettings

__all__ = ["Settings", "CameraSyncService", "CameraConfiguration", "DetectionSettings"]
