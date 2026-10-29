"""Asynchronous OCR of unresolved tracks, using their exact detection frames."""
from collections import OrderedDict
from dataclasses import dataclass, replace
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
    """One OCR worker; latest frames only and no repeated OCR for known tracks.

    The passage worker supplies immutable frame-to-track associations after
    making the vehicle overlay available. No tracker objects cross threads.
    """
    def __init__(self, sync_service, detection_runtime, settings, *, track_source=None,
                 pipeline_factory=RecognitionPipeline):
        self.sync, self.detection, self.settings = sync_service, detection_runtime, settings
        self.track_source, self.pipeline_factory = track_source, pipeline_factory
        self.pipeline = None
        self._lock = threading.Lock()
        self._results, self._pending = {}, {}
        self._known, self._attempted, self._configs = {}, {}, {}
        self._seen, self._next_at, self._log_at = {}, {}, {}
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, name="plate-recognition", daemon=True)
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("This recognition runtime has already been started")
        if self.track_source is None:
            raise ValueError("Recognition requires the shared vehicle tracker")
        self._started = True
        self.pipeline = self.pipeline_factory(self.detection.settings, self.settings)
        self.track_source.subscribe_updates(self._wake)
        self._thread.start()
        LOGGER.info("Plate recognition worker started interval_seconds=%.2f max_vehicles=%d reuse_track_plate=true",
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
        current = next((c for c in self.sync.snapshot().cameras if c.camera.code == camera_code), None)
        if current != config or time.monotonic()-result.received_monotonic > self.settings.max_result_age:
            return None
        return result

    def drain(self, camera_code):
        """Hand off all completed track reads; a later frame cannot overwrite them."""
        current = next((c for c in self.sync.snapshot().cameras if c.camera.code == camera_code), None)
        now = time.monotonic()
        with self._lock:
            entries = self._pending.pop(camera_code, {}).values()
        return tuple(result for config, result in entries
                     if config == current and not self._stop.is_set()
                     and (result.vehicles[0].license_plate.text
                          or now-result.received_monotonic <= self.settings.max_result_age))

    def _reconcile(self, configs):
        with self._lock:
            for code in set(self._configs) | set(configs):
                if self._configs.get(code) != configs.get(code):
                    self._results.pop(code, None)
                    self._pending.pop(code, None)
                    self._known.pop(code, None)
                    self._attempted.pop(code, None)
                    for cache in (self._seen, self._next_at, self._log_at):
                        cache.pop(code, None)
        self._configs = configs

    def _process_camera(self, code, config):
        now = time.monotonic()
        if now < self._next_at.get(code, 0.):
            return
        detected = self.detection.latest(code)
        if detected is None or detected.camera != config.camera:
            return
        selection = self.track_source.recognition_targets(detected)
        if selection is None:
            return  # Tracking/fast overlay has not consumed this exact frame yet.
        targets, active = selection
        known = self._known.setdefault(code, set())
        known.intersection_update(active)
        attempted = self._attempted.setdefault(code, {})
        for identity in tuple(attempted):
            if identity not in active:
                del attempted[identity]
        with self._lock:
            pending = self._pending.setdefault(code, OrderedDict())
            for identity, (_, result) in tuple(pending.items()):
                if (not result.vehicles[0].license_plate.text
                        and now-result.received_monotonic > self.settings.max_result_age):
                    del pending[identity]
            available = self.settings.max_pending_results-len(pending)
        if available <= 0:
            return  # Bound results without losing a completed plate to eviction.
        key = (config, detected.frame.sequence, detected.frame.received_monotonic)
        if self._seen.get(code) == key:
            return
        targets = {i: identity for i, identity in targets if identity not in known}
        if not targets:
            self._seen[code] = key
            return
        indices = sorted(targets, key=lambda i: (attempted.get(targets[i], 0.), -detected.detections[i].confidence))
        indices = indices[:min(self.settings.max_vehicles, available)]
        current = next((c for c in self.sync.snapshot().cameras if c.camera.code == code), None)
        if current != config:
            return
        self._seen[code] = key
        self._next_at[code] = now+self.settings.interval_seconds
        started = time.monotonic()
        completed = []
        for index in indices:
            current = next((c for c in self.sync.snapshot().cameras if c.camera.code == code), None)
            if (self._stop.is_set() or current != config
                    or time.monotonic()-detected.frame.received_monotonic > self.settings.max_result_age):
                break
            attempted[targets[index]] = time.monotonic()
            reads = self.pipeline.process(detected, config.zones, detection_indices=frozenset({index}))
            elapsed = (time.monotonic()-started)*1000
            vehicles = tuple(replace(read, object_key=targets[index]) for read in reads if read.detection_index == index)
            result = RecognitionResult(config.camera, detected.frame.sequence, detected.frame.captured_at,
                                       detected.frame.received_monotonic, vehicles, elapsed)
            current = next((c for c in self.sync.snapshot().cameras if c.camera.code == code), None)
            ready = (not self._stop.is_set() and current == config
                     and time.monotonic()-result.received_monotonic <= self.settings.max_result_age)
            if not ready:
                break
            # Commit each vehicle immediately; a later slow crop must not lose
            # an already completed plate or force it through OCR again.
            with self._lock:
                self._results[code] = (config, result)
                pending = self._pending.setdefault(code, OrderedDict())
                for read in vehicles:
                    pending[read.object_key] = (config, replace(result, vehicles=(read,)))
                    if read.license_plate.text:
                        known.add(read.object_key)
            completed.extend(vehicles)
            self.track_source.wake()
        elapsed = (time.monotonic()-started)*1000
        age = time.monotonic()-detected.frame.received_monotonic
        if time.monotonic() >= self._log_at.get(code, 0.):
            LOGGER.info("Plate recognition camera=%s sequence=%d vehicles=%d recognized=%d cached_tracks=%d "
                        "inference_ms=%.1f frame_age_ms=%.1f result=%s", code, detected.frame.sequence,
                        len(completed), sum(bool(v.license_plate.text) for v in completed), len(known),
                        elapsed, age*1000, "READY" if completed else "NO_FRESH_READ")
            self._log_at[code] = time.monotonic()+10.

    def _run(self):
        while not self._stop.is_set():
            self._wake.clear()
            configs = {c.camera.code: c for c in self.sync.snapshot().cameras}
            self._reconcile(configs)
            for code, config in configs.items():
                if self._stop.is_set():
                    break
                try:
                    self._process_camera(code, config)
                except Exception as error:
                    LOGGER.error("Plate recognition failed camera=%s type=%s", code, type(error).__name__)
                    self._next_at[code] = time.monotonic()+2.
            self._wake.wait(.02)

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self.track_source is not None:
            self.track_source.unsubscribe_updates(self._wake)
        if self._thread.ident is not None:
            self._thread.join(self.detection.settings.shutdown_timeout_seconds)
        stopped = not self._thread.is_alive()
        if stopped:
            with self._lock:
                self._results.clear()
                self._pending.clear()
                self._known.clear()
                self._attempted.clear()
        else:
            LOGGER.error("Recognition shutdown exceeded timeout; inference remains")
        return stopped
