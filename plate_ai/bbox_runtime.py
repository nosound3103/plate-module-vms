"""Publish the newest tracked overlay at a bounded per-camera cadence."""
import logging
import threading
import time
from .bbox import build_bbox_message

LOGGER = logging.getLogger(__name__)


class BboxRuntime:
    def __init__(self, sync_service, passage_runtime, settings, *, publish=None):
        self.sync, self.passages, self.settings = sync_service, passage_runtime, settings
        self.transport = None
        if publish is None:
            from .bbox_mqtt import BboxMqttPublisher
            self.transport = BboxMqttPublisher(sync_service.settings)
            publish = self.transport.publish
        self.publish = publish
        self._next_at, self._last, self._visible = {}, {}, {}
        self._cursor = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="plate-bbox", daemon=True)
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("This bbox runtime has already been started")
        self._started = True
        try:
            if self.transport is not None:
                self.transport.start()
            self._thread.start()
        except Exception:
            self.stop()
            raise
        LOGGER.info("PLATE bbox worker started max_fps=%.1f max_frame_age=%.2f topic_prefix=%s",
                    self.settings.max_fps, self.settings.max_frame_age, self.settings.topic_prefix)
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def _send(self, snapshot, *, clear=False):
        current = next((c.camera for c in self.sync.snapshot().cameras
                        if c.camera.code == snapshot.camera.code), None)
        if not clear and current != snapshot.camera:
            return
        message = build_bbox_message(snapshot, self.sync.settings.organization_id, self.settings, clear=clear)
        queued = self.publish(message, topic_prefix=self.settings.topic_prefix)
        if queued:
            self._visible[snapshot.camera.code] = bool(message["detections"])

    def _run(self):
        while not self._stop.is_set():
            try:
                configs = {c.camera.code: c.camera for c in self.sync.snapshot().cameras}
                for code in tuple(self._last):
                    if configs.get(code) != self._last[code].camera:
                        if self._visible.get(code):
                            self._send(self._last[code], clear=True)
                        self._last.pop(code, None)
                        self._visible.pop(code, None)
                        self._next_at.pop(code, None)
                codes = list(configs)
                if codes:
                    # Rotate the first camera so a busy MQTT socket cannot
                    # repeatedly favor the same camera on every scheduling turn.
                    start = self._cursor % len(codes)
                    due = any(time.monotonic() >= self._next_at.get(code, 0.) for code in codes)
                    codes = codes[start:]+codes[:start]
                    if due:
                        self._cursor = (start+1) % len(codes)
                for code in codes:
                    if self._stop.is_set():
                        break
                    now = time.monotonic()
                    if now < self._next_at.get(code, 0.):
                        continue
                    self._next_at[code] = now+1/self.settings.max_fps
                    try:
                        snapshot = self.passages.latest_overlay(code)
                        if (snapshot is not None and snapshot.camera == configs[code]
                                and now-snapshot.received_monotonic <= self.settings.max_frame_age):
                            self._last[code] = snapshot
                            self._send(snapshot)
                        elif self._visible.get(code) and code in self._last:
                            self._send(self._last[code], clear=True)
                    except Exception as error:
                        LOGGER.error("PLATE bbox failed camera=%s type=%s", code, type(error).__name__)
                for code in tuple(self._next_at):
                    if code not in configs:
                        self._next_at.pop(code, None)
            except Exception as error:
                LOGGER.error("PLATE bbox controller failed type=%s", type(error).__name__)
            self._stop.wait(.01)

    def stop(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(self.passages.detection.settings.shutdown_timeout_seconds)
        stopped = not self._thread.is_alive()
        if self.transport is not None:
            self.transport.stop()
        if stopped:
            self._last.clear()
            self._visible.clear()
            self._next_at.clear()
        else:
            LOGGER.error("PLATE bbox shutdown exceeded timeout")
        return stopped
