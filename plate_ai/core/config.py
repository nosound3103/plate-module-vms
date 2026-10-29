"""Validated process configuration; this module never opens .env files."""
from dataclasses import dataclass, field
import math
import os
import re


@dataclass(frozen=True)
class Settings:
    broker_host: str = field(repr=False)
    username: str = field(repr=False)
    password: str = field(repr=False)
    organization_id: int
    zone_ids: tuple[int, ...]
    client_id: str
    broker_port: int = 443
    websocket_path: str = "/mqtt"
    camera_code: str | None = None
    camera_sync_seconds: float = 5.
    keepalive_seconds: int = 30
    reconnect_max_seconds: int = 30
    config_queue_size: int = 256
    config_max_bytes: int = 1048576
    shutdown_timeout_seconds: float = 10.
    log_level: str = "INFO"

    def __post_init__(self):
        if not self.broker_host or any(c in self.broker_host for c in "/:") or any(c.isspace() for c in self.broker_host):
            raise ValueError("MQTT_HOST must be a hostname without a scheme or path")
        if not self.username or not self.password:
            raise ValueError("MQTT_USERNAME and MQTT_PASSWORD are required")
        if self.client_id.lower().startswith("vms_staging_backend"):
            raise ValueError("MQTT_CLIENT_ID must not start with vms_staging_backend")
        if not re.fullmatch(r"tts_[A-Za-z0-9][A-Za-z0-9_-]*_01", self.client_id):
            raise ValueError("MQTT_CLIENT_ID must follow tts_<name>_01; name must start with a letter or digit and contain only letters, digits, underscores or hyphens")
        if self.organization_id <= 0 or not self.zone_ids or any(z <= 0 for z in self.zone_ids):
            raise ValueError("VMS_ORGANIZATION_ID and VMS_ZONE_IDS must contain positive IDs")
        if not 1 <= self.broker_port <= 65535:
            raise ValueError("MQTT_PORT must be between 1 and 65535")
        if not self.websocket_path.startswith("/") or any(c.isspace() for c in self.websocket_path):
            raise ValueError("MQTT_WS_PATH must start with / and contain no whitespace")
        for name in ("camera_sync_seconds", "shutdown_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("keepalive_seconds", "reconnect_max_seconds", "config_queue_size", "config_max_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.camera_code and any(c in self.camera_code for c in "/+#\x00"):
            raise ValueError("VMS_CAMERA_CODE must be a single MQTT topic segment")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")

    @property
    def bbox_client_id(self):
        return f"{self.client_id[:-3]}_bbox_01"

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ

        def read(name, *, default=None, strip=True):
            value = env.get(name)
            if value is not None and value.strip():
                return value.strip() if strip else value
            if default is not None:
                return default
            raise ValueError(f"Missing required environment variable: {name}")

        def number(name, convert, *, default):
            try:
                return convert(read(name, default=default))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric environment variable: {name}") from None

        raw_zones = read("VMS_ZONE_IDS")
        try:
            zone_ids = tuple(dict.fromkeys(int(z.strip()) for z in raw_zones.split(",")))
        except ValueError:
            raise ValueError("VMS_ZONE_IDS must be comma-separated positive integers") from None
        return cls(
            broker_host=read("MQTT_HOST"),
            username=read("MQTT_USERNAME"),
            password=read("MQTT_PASSWORD", strip=False),
            organization_id=number("VMS_ORGANIZATION_ID", int, default="0"),
            zone_ids=zone_ids,
            client_id=read("MQTT_CLIENT_ID"),
            broker_port=number("MQTT_PORT", int, default="443"),
            websocket_path=read("MQTT_WS_PATH", default="/mqtt"),
            camera_code=read("VMS_CAMERA_CODE", default="") or None,
            camera_sync_seconds=number("VMS_CAMERA_SYNC_SECONDS", float, default="5"),
            keepalive_seconds=number("MQTT_KEEPALIVE_SECONDS", int, default="30"),
            reconnect_max_seconds=number("MQTT_RECONNECT_MAX_SECONDS", int, default="30"),
            config_queue_size=number("VMS_CONFIG_QUEUE_SIZE", int, default="256"),
            config_max_bytes=number("VMS_CONFIG_MAX_BYTES", int, default="1048576"),
            shutdown_timeout_seconds=number("VMS_SHUTDOWN_TIMEOUT_SECONDS", float, default="10"),
            log_level=read("LOG_LEVEL", default="INFO").upper(),
        )
