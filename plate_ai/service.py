"""Camera/zone sync controller; no RTSP, model loading or AI publishing."""
from dataclasses import dataclass
import json
import logging
import queue
import threading
import time
from .domain import Camera, Zone, CrossingLine, ZoneStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfiguration:
    camera: Camera
    zones: tuple[Zone | CrossingLine, ...]


@dataclass(frozen=True)
class SyncSnapshot:
    revision: int = 0
    cameras: tuple[CameraConfiguration, ...] = ()


class CameraSyncService:
    """Call start/stop or use a context manager; snapshot() is safe for host threads.

    Transport is injectable through set_handlers/start/stop/request_cameras/
    subscribe_camera/unsubscribe_camera. All mutable registry state belongs to
    one controller thread. Public snapshots contain immutable domain objects.
    """
    def __init__(self, settings, *, transport=None):
        self.settings = settings
        if transport is None:
            from .mqtt import MqttTransport
            transport = MqttTransport(settings)
        self.transport = transport
        self._snapshots = {}
        self._cameras = {}
        self._zones = {}
        self._inbox = queue.Queue(maxsize=settings.config_queue_size)
        self._stop = threading.Event()
        self._resync = threading.Event()
        self._thread = None
        self._state_lock = threading.Lock()
        self._state = SyncSnapshot()
        self.transport.set_handlers(self.submit_message, self._connected)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("This service instance has already been started")
        self._thread = threading.Thread(target=self._run, name="camera-sync", daemon=False)
        self._thread.start()
        try:
            self.transport.start()
        except Exception:
            self.stop()
            raise
        LOGGER.info("Camera sync started organization=%d zones=%d", self.settings.organization_id,
                    len(self.settings.zone_ids))
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(self.settings.shutdown_timeout_seconds)
            if self._thread.is_alive():
                LOGGER.error("Camera sync controller exceeded shutdown timeout")
        self.transport.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def snapshot(self):
        with self._state_lock:
            return self._state

    def submit_message(self, topic, payload):
        if self._stop.is_set():
            return
        if len(payload) > self.settings.config_max_bytes:
            LOGGER.warning("Configuration rejected result=PAYLOAD_TOO_LARGE")
            return
        try:
            self._inbox.put_nowait((topic, payload))
        except queue.Full:
            self._resync.set()
            LOGGER.warning("Configuration queue full; discovery will be refreshed")

    def _connected(self):
        self.submit_message(None, b"")

    def _run(self):
        next_sync = 0.
        while not self._stop.is_set():
            try:
                topic, payload = self._inbox.get(timeout=.25)
                if topic is None:
                    self._reset_discovery()
                    next_sync = 0.
                else:
                    self._handle_message(topic, payload)
            except queue.Empty:
                pass
            except (ValueError, TypeError, KeyError, AttributeError, UnicodeDecodeError, OverflowError):
                # Camera payloads contain RTSP credentials. Never print data or
                # exception messages that could include an offending value.
                LOGGER.warning("Configuration rejected result=INVALID_PAYLOAD")
            except Exception as error:
                LOGGER.error("Configuration processing failed type=%s", type(error).__name__)
            if self._resync.is_set():
                self._resync.clear()
                # Discard partial old updates before requesting complete state.
                while True:
                    try:
                        self._inbox.get_nowait()
                    except queue.Empty:
                        break
                self._reset_discovery()
                next_sync = 0.
            now = time.monotonic()
            if now >= next_sync:
                try:
                    self.transport.request_cameras()
                except Exception as error:
                    LOGGER.error("Camera sync request failed type=%s", type(error).__name__)
                next_sync = now + self.settings.camera_sync_seconds

    def _reset_discovery(self):
        for code in tuple(self._cameras):
            self.transport.unsubscribe_camera(code)
        self._snapshots.clear()
        self._cameras.clear()
        self._zones.clear()
        self._publish_state()

    def _handle_message(self, topic, raw):
        prefix = "smart_vms/cameras/zone/"
        if topic.startswith(prefix):
            zone_id = int(topic[len(prefix):])
            if zone_id not in self.settings.zone_ids:
                return
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("cameras"), list):
                raise ValueError("Expected full camera snapshot")
            if int(data.get("zone_id", zone_id)) != zone_id:
                raise ValueError("Zone mismatch")
            if "organization_id" in data and int(data["organization_id"]) != self.settings.organization_id:
                raise ValueError("Organization mismatch")
            if int(data.get("total", len(data["cameras"]))) != len(data["cameras"]):
                raise ValueError("Partial snapshots are unsupported")
            snapshot = {}
            for entry in data["cameras"]:
                camera = Camera.parse(entry)
                if camera is not None and (not self.settings.camera_code or camera.code == self.settings.camera_code):
                    if camera.code in snapshot:
                        raise ValueError("Duplicate camera code")
                    snapshot[camera.code] = camera
            # Process the refreshed zone last so a changed RTSP URL for the
            # same camera UUID can propagate across overlapping zone snapshots.
            proposed = {key: value for key, value in self._snapshots.items() if key != zone_id}
            proposed[zone_id] = snapshot
            desired = {}
            for entries in proposed.values():
                for code, camera in entries.items():
                    if code in desired and desired[code].id != camera.id:
                        raise ValueError("Conflicting camera UUIDs across zones")
                    desired[code] = camera
            self._snapshots = proposed
            self._reconcile(desired)
            LOGGER.info("Synchronized PLATE cameras zone=%d count=%d total_cameras=%d",
                        zone_id, len(snapshot), len(desired))
            return
        parts = topic.split("/")
        if len(parts) < 4 or parts[:2] != ["smart_vms", "cameras"] or parts[3] != "ai_zones":
            return
        code = parts[2]
        if code not in self._zones:
            return
        data = json.loads(raw.decode("utf-8")) if raw else None
        suffix = "/".join(parts[4:])
        if isinstance(data, dict):
            if data.get("camera_code", code) != code or data.get("camera_id", self._cameras[code].id) != self._cameras[code].id:
                raise ValueError("Camera mismatch")
        self._zones[code].apply(suffix, data)
        self._publish_state()
        zones = self._zones[code].zones
        lines = sum(isinstance(z, CrossingLine) for z in zones)
        LOGGER.info("PLATE configuration camera=%s polygons=%d lines=%d", code, len(zones)-lines, lines)

    def _reconcile(self, desired):
        for code, previous in tuple(self._cameras.items()):
            current = desired.get(code)
            if current is None or current.id != previous.id:
                self.transport.unsubscribe_camera(code)
                self._zones.pop(code, None)
        self._cameras = desired
        for code in desired:
            if code not in self._zones:
                self._zones[code] = ZoneStore()
                self.transport.subscribe_camera(code)
        self._publish_state()

    def _publish_state(self):
        cameras = tuple(CameraConfiguration(self._cameras[code], self._zones[code].zones)
                        for code in sorted(self._cameras))
        with self._state_lock:
            if cameras != self._state.cameras:
                self._state = SyncSnapshot(self._state.revision+1, cameras)
