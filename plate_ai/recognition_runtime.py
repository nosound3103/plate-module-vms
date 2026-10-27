"""Bounded asynchronous recognition over the latest vehicle detection results."""
from dataclasses import dataclass
import logging
import threading
import time
from .domain import Camera
from .recognition import RecognitionPipeline, VehicleRecognition

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecognitionResult:
    camera: Camera
    sequence: int
    captured_at: float
    received_monotonic: float
    vehicles: tuple[VehicleRecognition, ...]
    inference_ms: float


class RecognitionRuntime:
    """One OCR worker shared across cameras, without an accumulating frame queue.

    Detection keeps running independently. Each turn reads the newest available
    detection result, with crops taken from exactly that result's source frame.
    """
    def __init__(self, sync_service, detection_runtime, settings, *, pipeline_factory=RecognitionPipeline):
        self.sync, self.detection, self.settings = sync_service, detection_runtime, settings
        self.pipeline_factory = pipeline_factory
        self.pipeline = None
        self._lock = threading.Lock()
        self._results, self._seen, self._next_at, self._log_at = {}, {}, {}, {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="plate-recognition", daemon=True)
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("This recognition runtime has already been started")
        self._started = True
        self.pipeline = self.pipeline_factory(self.detection.settings, self.settings)
        self._thread.start()
        LOGGER.info("Plate recognition worker started interval_seconds=%.2f max_vehicles=%d",
                    self.settings.interval_seconds, self.settings.max_vehicles)
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def latest(self, camera_code):
        with self._lock:
            entry = self._results.get(camera_code)
        if entry is None or self._stop.is_set():
            return None
        config, result = entry
        if time.monotonic()-result.received_monotonic > self.settings.max_result_age:
            return None
        current = next((c for c in self.sync.snapshot().cameras if c.camera.code == camera_code), None)
        detected = self.detection.latest(camera_code)
        if current != config or detected is None or detected.camera != result.camera:
            return None
        return result

    def _run(self):
        while not self._stop.is_set():
            configs = {c.camera.code: c for c in self.sync.snapshot().cameras}
            with self._lock:
                for code, (config, _) in tuple(self._results.items()):
                    if configs.get(code) != config:
                        del self._results[code]
            for cache in (self._seen, self._next_at, self._log_at):
                for code in tuple(cache):
                    if code not in configs:
                        del cache[code]
            for code, config in configs.items():
                if self._stop.is_set():
                    break
                if time.monotonic() < self._next_at.get(code, 0.):
                    continue
                detected = self.detection.latest(code)
                if detected is None or detected.camera != config.camera:
                    continue
                key = (config, detected.frame.sequence, detected.frame.received_monotonic)
                if self._seen.get(code) == key:
                    continue
                self._seen[code] = key
                self._next_at[code] = time.monotonic()+self.settings.interval_seconds
                try:
                    started = time.monotonic()
                    vehicles = self.pipeline.process(detected, config.zones)
                    elapsed = (time.monotonic()-started)*1000
                    result = RecognitionResult(config.camera, detected.frame.sequence, detected.frame.captured_at,
                                               detected.frame.received_monotonic, vehicles, elapsed)
                    current = next((c for c in self.sync.snapshot().cameras if c.camera.code == code), None)
                    live = self.detection.latest(code)
                    age = time.monotonic()-result.received_monotonic
                    ready = (not self._stop.is_set() and current == config and live is not None
                             and live.camera == config.camera and age <= self.settings.max_result_age)
                    with self._lock:
                        if ready:
                            self._results[code] = (config, result)
                        else:
                            self._results.pop(code, None)
                    if time.monotonic() >= self._log_at.get(code, 0.):
                        LOGGER.info("Plate recognition camera=%s sequence=%d vehicles=%d plates=%d recognized=%d "
                                    "inference_ms=%.1f frame_age_ms=%.1f result=%s", code, result.sequence,
                                    len(vehicles), sum(v.license_plate.bbox is not None for v in vehicles),
                                    sum(bool(v.license_plate.text) for v in vehicles), elapsed, age*1000,
                                    "READY" if ready else "STALE")
                        self._log_at[code] = time.monotonic()+10.
                except Exception as error:
                    with self._lock:
                        self._results.pop(code, None)
                    LOGGER.error("Plate recognition failed camera=%s type=%s", code, type(error).__name__)
                    self._next_at[code] = time.monotonic()+2.
            self._stop.wait(.02)

    def stop(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(self.detection.settings.shutdown_timeout_seconds)
        with self._lock:
            self._results.clear()
        stopped = not self._thread.is_alive()
        if not stopped:
            LOGGER.error("Recognition shutdown exceeded timeout; inference remains")
        return stopped
