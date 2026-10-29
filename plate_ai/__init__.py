"""PLATE integration foundation; importing it opens no network connections."""
from plate_ai.core.config import Settings
from plate_ai.camera.sync import CameraSyncService, CameraConfiguration
from plate_ai.detection.config import DetectionSettings

__all__ = ["Settings", "CameraSyncService", "CameraConfiguration", "DetectionSettings"]
