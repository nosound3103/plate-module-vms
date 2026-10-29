"""Camera lifecycle supervision and shared round-robin vehicle inference."""
from dataclasses import dataclass
from collections import deque
import logging
import os
import threading
import time
from .detection import VehicleDetection, VehicleDetector
from .domain import Camera
from .rtsp import CapturedFrame, LatestFrameStream

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    camera: Camera
    frame: CapturedFrame
    detections: tuple[VehicleDetection, ...]
    inference_ms: float
    completed_monotonic: float = 0.


@dataclass(frozen=True)
class DetectionMetadata:
    camera: Camera
    sequence: int
    captured_at: float
    received_monotonic: float
    width: int
    height: int
    detections: tuple[VehicleDetection, ...]
    completed_monotonic: float = 0.


@dataclass
class _CameraState:
    reader: LatestFrameStream
    next_inference_at: float = 0.
    next_log_at: float = 0.


class DetectionRuntime:
    """One model/inference thread and one reader per active camera.

    Registry updates never wait for inference or RTSP I/O. Results retain only
    the latest frame per camera, preserving matching crops for downstream stages.
    """
    def __init__(self, sync_service, settings, *, detector_factory=VehicleDetector,
                 stream_factory=LatestFrameStream, metadata_frames=0, metadata_seconds=4., evidence_frames=0):
        if metadata_frames < 0 or metadata_seconds <= 0:
            raise ValueError("Invalid detection metadata limits")
        self.sync, self.settings = sync_service, settings
        self.detector_factory, self.stream_factory = detector_factory, stream_factory
        self.detector = None
        self._lock = threading.Lock()
        self._states, self._retiring, self._results = {}, {}, {}
        self._metadata = {}
        self._subscribers = set()
        self._frames = {}
        self._evidence_frames = evidence_frames
        self._metadata_frames, self._metadata_seconds = metadata_frames, metadata_seconds
        self._stop = threading.Event()
        self._controller = threading.Thread(target=self._supervise, name="detection-controller", daemon=True)
        self._inference = threading.Thread(target=self._infer, name="vehicle-inference", daemon=True)
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("This detection runtime has already been started")
        self._started = True
        # Configure video logging before OpenCV is imported by the detector.
        os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                              f"rtsp_transport;{self.settings.rtsp_transport}|fflags;nobuffer|flags;low_delay")
        import torch
        torch.set_num_threads(self.settings.torch_threads)
        try:
            self.detector = self.detector_factory(self.settings)
        except Exception as error:
            LOGGER.error("Vehicle model initialization failed type=%s; check device, weight and checksum",
                         type(error).__name__)
            raise
        self._controller.start()
        try:
            self._inference.start()
        except Exception:
            self.stop()
            raise
        LOGGER.info("Detection runtime started device=%s max_fps=%.1f", self.settings.device, self.settings.max_fps)
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def latest(self, camera_code):
        with self._lock:
            result = self._results.get(camera_code)
        current = next((c.camera for c in self.sync.snapshot().cameras if c.camera.code == camera_code), None)
        if (result is not None and result.camera == current and not self._stop.is_set()
                and time.monotonic()-result.frame.received_monotonic <= self.settings.max_frame_age):
            return result
        return None

    def metadata(self, camera_code):
        """Recent detection geometry for asynchronous association; no frame images."""
        now = time.monotonic()
        current = next((c.camera for c in self.sync.snapshot().cameras if c.camera.code == camera_code), None)
        with self._lock:
            return tuple(m for m in self._metadata.get(camera_code, ())
                         if m.camera == current and not self._stop.is_set()
                         and now-m.received_monotonic <= self._metadata_seconds)

    @property
    def metadata_enabled(self):
        return self._metadata_frames > 0

    def subscribe_updates(self, event):
        with self._lock:
            self._subscribers.add(event)
        event.set()

    def unsubscribe_updates(self, event):
        with self._lock:
            self._subscribers.discard(event)

    def frame_for(self, metadata):
        """Exact sequence only; never substitute a newer frame for evidence/HSV."""
        with self._lock:
            return next((f for f in self._frames.get(metadata.camera.code, ())
                         if f.sequence == metadata.sequence
                         and f.received_monotonic == metadata.received_monotonic), None)

    def _supervise(self):
        while not self._stop.is_set():
            desired = {c.camera.code: c.camera for c in self.sync.snapshot().cameras}
            try:
                with self._lock:
                    for code, state in tuple(self._states.items()):
                        if desired.get(code) != state.reader.camera or not state.reader.is_alive():
                            state.reader.request_stop()
                            self._retiring[code] = state.reader
                            del self._states[code]
                            self._results.pop(code, None)
                            self._metadata.pop(code, None)
                            self._frames.pop(code, None)
                            LOGGER.info("Detection camera removed camera=%s", code)
                    for code, reader in tuple(self._retiring.items()):
                        if not reader.is_alive():
                            del self._retiring[code]
                    for code, camera in desired.items():
                        if code not in self._states and code not in self._retiring:
                            reader = self.stream_factory(camera, self.settings)
                            reader.start()
                            self._states[code] = _CameraState(reader)
                            LOGGER.info("Detection camera started camera=%s", code)
            except Exception as error:
                LOGGER.error("Detection reconciliation failed type=%s", type(error).__name__)
                self._stop.wait(1.)
            self._stop.wait(.1)

    def _infer(self):
        while not self._stop.is_set():
            with self._lock:
                states = tuple(self._states.items())
            for code, state in states:
                if self._stop.is_set():
                    break
                now = time.monotonic()
                if now < state.next_inference_at:
                    continue
                frame = state.reader.latest()
                if frame is None or now-frame.received_monotonic > self.settings.max_frame_age:
                    continue
                with self._lock:
                    if self._states.get(code) is not state:
                        continue
                current = next((c.camera for c in self.sync.snapshot().cameras if c.camera.code == code), None)
                if current != state.reader.camera:
                    continue
                state.next_inference_at = now+1/self.settings.max_fps
                try:
                    started = time.monotonic()
                    detections = self.detector.detect(frame.image)
                    completed = time.monotonic()
                    inference_ms = (completed-started)*1000
                    result = DetectionResult(state.reader.camera, frame, tuple(detections), inference_ms, completed)
                    age = time.monotonic()-frame.received_monotonic
                    current = next((c.camera for c in self.sync.snapshot().cameras if c.camera.code == code), None)
                    with self._lock:
                        if (self._stop.is_set() or self._states.get(code) is not state
                                or current != state.reader.camera):
                            continue
                        if age <= self.settings.max_frame_age:
                            self._results[code] = result
                            if self._evidence_frames:
                                self._frames.setdefault(code, deque(maxlen=self._evidence_frames)).append(frame)
                            if self._metadata_frames:
                                history = self._metadata.setdefault(code, deque(maxlen=self._metadata_frames))
                                history.append(DetectionMetadata(result.camera, frame.sequence, frame.captured_at,
                                                                 frame.received_monotonic, frame.image.shape[1],
                                                                 frame.image.shape[0], result.detections, completed))
                                while history and time.monotonic()-history[0].received_monotonic > self._metadata_seconds:
                                    history.popleft()
                        else:
                            self._results.pop(code, None)
                        subscribers = tuple(self._subscribers) if age <= self.settings.max_frame_age else ()
                    for event in subscribers:
                        event.set()
                    if time.monotonic() >= state.next_log_at:
                        LOGGER.info("Vehicle detection camera=%s sequence=%d count=%d inference_ms=%.1f frame_age_ms=%.1f result=%s",
                                    code, frame.sequence, len(detections), inference_ms, age*1000,
                                    "READY" if age <= self.settings.max_frame_age else "STALE")
                        state.next_log_at = time.monotonic()+10.
                except Exception as error:
                    with self._lock:
                        self._results.pop(code, None)
                    LOGGER.error("Vehicle inference failed camera=%s type=%s", code, type(error).__name__)
                    state.next_inference_at = time.monotonic()+2.
            self._stop.wait(.01)

    def stop(self):
        self._stop.set()
        deadline = time.monotonic()+self.settings.shutdown_timeout_seconds
        if self._controller.ident is not None:
            self._controller.join(max(0., deadline-time.monotonic()))
        with self._lock:
            readers = tuple(s.reader for s in self._states.values())+tuple(self._retiring.values())
            self._states.clear()
            self._results.clear()
            self._metadata.clear()
            self._frames.clear()
            self._retiring.clear()
        for reader in readers:
            reader.request_stop()
        if self._inference.ident is not None:
            self._inference.join(max(0., deadline-time.monotonic()))
        stopped = not self._controller.is_alive() and not self._inference.is_alive()
        for reader in readers:
            stopped = reader.stop(max(0., deadline-time.monotonic())) and stopped
        if not stopped:
            LOGGER.error("Detection shutdown exceeded timeout; pending I/O or inference remains")
        return stopped
