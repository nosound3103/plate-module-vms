"""Bounded RTSP acquisition: one newest frame per camera."""
from dataclasses import dataclass, field
import logging
import queue
import threading
import time

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    captured_at: float
    received_monotonic: float
    image: object = field(repr=False, compare=False)


class LatestFrameStream:
    def __init__(self, camera, settings):
        self.camera, self.settings = camera, settings
        self.frames = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, name=f"rtsp-{camera.code}", daemon=True)

    def start(self):
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def request_stop(self):
        self._stop.set()

    def stop(self, timeout=0.):
        self.request_stop()
        if self._thread.ident is not None:
            self._thread.join(timeout)
        return not self.is_alive()

    def latest(self):
        try:
            return self.frames.get_nowait()
        except queue.Empty:
            return None

    def _read(self):
        import cv2
        delay, sequence = 1., 0
        while not self._stop.is_set():
            capture = None
            try:
                capture = cv2.VideoCapture(self.camera.source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.settings.open_timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.settings.read_timeout_ms,
                ])
                if not capture.isOpened():
                    raise RuntimeError("RTSP connection unavailable")
                LOGGER.info("RTSP connected camera=%s", self.camera.code)
                while not self._stop.is_set():
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError("RTSP frame unavailable")
                    sequence += 1
                    frame = CapturedFrame(sequence, time.time(), time.monotonic(), image)
                    delay = 1.
                    self.latest()
                    self.frames.put_nowait(frame)
            except Exception as error:
                self.latest()
                if not self._stop.is_set():
                    LOGGER.warning("RTSP unavailable camera=%s type=%s reconnect_seconds=%.1f",
                                   self.camera.code, type(error).__name__, delay)
            finally:
                if capture is not None:
                    capture.release()
            self._stop.wait(delay)
            delay = min(delay*2, 30.)
