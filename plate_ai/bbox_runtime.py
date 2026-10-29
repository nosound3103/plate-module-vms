"""Wake on tracked detections and publish overlays without waiting for OCR."""
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
        self._budget, self._refill_at, self._last, self._visible = {}, {}, {}, {}
        self._sent, self._log_at = {}, {}
        self._queued_at = {}
        self._cursor = 0
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, name="plate-bbox", daemon=True)
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("This bbox runtime has already been started")
        self._started = True
        try:
            if self.transport is not None:
                self.transport.start()
            self.passages.subscribe_updates(self._wake)
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
            return False
        message = build_bbox_message(snapshot, self.sync.settings.organization_id, self.settings, clear=clear)
        queued = self.publish(message, topic_prefix=self.settings.topic_prefix)
        if queued:
            self._visible[snapshot.camera.code] = bool(message["detections"])
            now = time.monotonic()
            if clear:
                LOGGER.warning("PLATE bbox camera=%s sequence=%d result=CLEARED age_ms=%.1f",
                               snapshot.camera.code, snapshot.sequence,
                               (now-snapshot.received_monotonic)*1000)
            else:
                previous = self._queued_at.get(snapshot.camera.code)
                if previous is not None and now-previous > self.settings.max_frame_age:
                    LOGGER.warning("PLATE bbox camera=%s sequence=%d result=RESUMED queue_gap_ms=%.1f",
                                   snapshot.camera.code, snapshot.sequence, (now-previous)*1000)
                self._queued_at[snapshot.camera.code] = now
            if not clear and now >= self._log_at.get(snapshot.camera.code, 0.):
                completed = snapshot.detection_completed_monotonic
                LOGGER.info("PLATE bbox camera=%s sequence=%d detection_to_queue_ms=%.1f frame_age_ms=%.1f result=QUEUED",
                            snapshot.camera.code, snapshot.sequence,
                            (now-completed)*1000 if completed else 0., message["frame_age_ms"])
                self._log_at[snapshot.camera.code] = now+10.
        return queued

    def _send_latest(self, snapshot):
        code, now = snapshot.camera.code, time.monotonic()
        previous = self._sent.get(code)
        if previous is snapshot:
            return False
        capacity = self.settings.burst_capacity
        budget = min(capacity, self._budget.get(code, capacity)
                     + max(0., now-self._refill_at.get(code, now))*self.settings.max_fps)
        self._budget[code], self._refill_at[code] = budget, now
        new_frame = (previous is None or (snapshot.sequence, snapshot.received_monotonic)
                     != (previous.sequence, previous.received_monotonic))
        # Enrichment may spend spare capacity, but must leave one packet for
        # the next detected frame. Rate is bounded; no queued stale frames.
        required = 1 if new_frame else 2
        if budget < required or not self._send(snapshot):
            return False
        self._budget[code] -= 1
        self._sent[code] = snapshot
        return True

    def _run(self):
        while not self._stop.is_set():
            self._wake.clear()
            try:
                configs = {c.camera.code: c.camera for c in self.sync.snapshot().cameras}
                for code in tuple(self._last):
                    if configs.get(code) != self._last[code].camera:
                        if self._visible.get(code):
                            self._send(self._last[code], clear=True)
                        self._last.pop(code, None)
                        self._visible.pop(code, None)
                        self._budget.pop(code, None)
                        self._refill_at.pop(code, None)
                        self._sent.pop(code, None)
                        self._log_at.pop(code, None)
                        self._queued_at.pop(code, None)
                codes = list(configs)
                if codes:
                    # Rotate the first camera so a busy MQTT socket cannot
                    # repeatedly favor the same camera on every scheduling turn.
                    start = self._cursor % len(codes)
                    codes = codes[start:]+codes[:start]
                    self._cursor = (start+1) % len(codes)
                for code in codes:
                    if self._stop.is_set():
                        break
                    now = time.monotonic()
                    try:
                        snapshot = self.passages.latest_overlay(code)
                        if (snapshot is not None and snapshot.camera == configs[code]
                                and now-snapshot.received_monotonic <= self.settings.max_frame_age):
                            self._last[code] = snapshot
                            self._send_latest(snapshot)
                        elif self._visible.get(code) and code in self._last:
                            self._send(self._last[code], clear=True)
                    except Exception as error:
                        LOGGER.error("PLATE bbox failed camera=%s type=%s", code, type(error).__name__)
            except Exception as error:
                LOGGER.error("PLATE bbox controller failed type=%s", type(error).__name__)
            self._wake.wait(.01)

    def stop(self):
        self._stop.set()
        self._wake.set()
        self.passages.unsubscribe_updates(self._wake)
        if self._thread.ident is not None:
            self._thread.join(self.passages.detection.settings.shutdown_timeout_seconds)
        stopped = not self._thread.is_alive()
        if self.transport is not None:
            self.transport.stop()
        if stopped:
            self._last.clear()
            self._visible.clear()
            self._budget.clear()
            self._refill_at.clear()
            self._sent.clear()
            self._log_at.clear()
            self._queued_at.clear()
        else:
            LOGGER.error("PLATE bbox shutdown exceeded timeout")
        return stopped
