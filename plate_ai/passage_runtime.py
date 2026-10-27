"""Track camera passages, associate asynchronous OCR and queue MQTT events."""
from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import threading
import time
from .domain import Camera, CrossingLine
from .events import build_plate_event
from .tracking import VehicleTracker, foot, intersects

LOGGER = logging.getLogger(__name__)


@dataclass
class _CameraState:
    camera: Camera
    tracker: VehicleTracker
    lines: tuple = ()
    last_frame_at: float = 0.
    lines_changed_at: float = 0.
    history: OrderedDict = field(default_factory=OrderedDict)
    last_ocr_key: tuple | None = None


class PassageRuntime:
    """One metadata controller; inference and network acknowledgements run separately."""
    def __init__(self, sync_service, detection_runtime, recognition_runtime, settings, *, publish=None):
        self.sync, self.detection, self.recognition = sync_service, detection_runtime, recognition_runtime
        self.settings = settings
        self.publish = publish if publish is not None else sync_service.transport.publish_plate_event
        self._states, self._pending = {}, {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="plate-passages", daemon=True)
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("This passage runtime has already been started")
        if not self.detection.metadata_enabled:
            raise ValueError("Passage runtime requires detection metadata history")
        self._started = True
        self._thread.start()
        LOGGER.info("PLATE passage worker started ocr_wait_seconds=%.1f max_pending=%d",
                    self.settings.ocr_wait_seconds, self.settings.max_pending)
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def _observe(self, state, metadata):
        visible = state.tracker.update(metadata)
        key = (metadata.sequence, metadata.received_monotonic)
        state.history[key] = visible
        state.last_frame_at = metadata.received_monotonic
        for track in visible.values():
            point = foot(track.box)
            for line in state.lines:
                distance = line.distance(point)
                if abs(distance) <= self.settings.crossing_deadband:
                    continue
                side = 1 if distance > 0 else -1
                previous = track.sides.get(line.id)
                if (previous and previous[0] != side and track.direction is None and track.hits >= 2
                        and intersects(previous[1], point, line)):
                    track.direction = "IN" if side == line.in_side else "OUT"
                    track.line_id = line.id
                    track.crossed_epoch = metadata.captured_at
                    track.crossed_monotonic = metadata.received_monotonic
                    if len(self._pending) < self.settings.max_pending:
                        self._pending[track.object_key] = (state.camera, track)
                        LOGGER.info("PLATE crossing camera=%s object_key=%s line=%s direction=%s result=PENDING_OCR",
                                    state.camera.code, track.object_key, line.id, track.direction)
                    else:
                        LOGGER.error("PLATE crossing camera=%s object_key=%s result=DROPPED_PENDING_FULL",
                                     state.camera.code, track.object_key)
                track.sides[line.id] = (side, point)

    def _recognize(self, state):
        result = self.recognition.latest(state.camera.code)
        if result is None or result.camera != state.camera:
            return
        key = (result.sequence, result.received_monotonic)
        if key == state.last_ocr_key:
            return
        state.last_ocr_key = key
        associations = state.history.get(key)
        if associations is None:
            LOGGER.debug("PLATE OCR association camera=%s sequence=%d result=NO_FRAME_HISTORY",
                         state.camera.code, result.sequence)
            return
        for read in result.vehicles:
            track = associations.get(read.detection_index)
            if track is not None:
                track.remember(read)

    def _flush(self, now):
        for key, (camera, track) in tuple(self._pending.items()):
            age = now-track.crossed_monotonic
            if age > self.settings.pending_ttl:
                LOGGER.error("PLATE event camera=%s object_key=%s result=DROPPED_EXPIRED", camera.code, key)
                del self._pending[key]
                continue
            if age < self.settings.ocr_wait_seconds or now < track.retry_at:
                continue
            if track.event is None:
                track.event = build_plate_event(camera, track)
            track.retry_at = now+self.settings.retry_seconds
            try:
                queued = self.publish(track.event, camera_code=camera.code, object_key=key)
            except Exception as error:
                LOGGER.error("PLATE event camera=%s object_key=%s result=PUBLISH_FAILED type=%s",
                             camera.code, key, type(error).__name__)
                continue
            if queued:
                track.sent = True
                del self._pending[key]

    def _run(self):
        while not self._stop.is_set():
            try:
                configs = {c.camera.code: c for c in self.sync.snapshot().cameras}
                for code, state in tuple(self._states.items()):
                    if code not in configs or configs[code].camera != state.camera:
                        del self._states[code]
                        LOGGER.info("PLATE tracking removed camera=%s", code)
                for code, config in configs.items():
                    try:
                        self._process_camera(code, config)
                    except Exception as error:
                        LOGGER.error("PLATE tracking failed camera=%s type=%s", code, type(error).__name__)
                self._flush(time.monotonic())
            except Exception as error:
                LOGGER.error("PLATE passage controller failed type=%s", type(error).__name__)
            self._stop.wait(.02)

    def _process_camera(self, code, config):
        now = time.monotonic()
        state = self._states.get(code)
        if state is None:
            state = _CameraState(config.camera, VehicleTracker(code, self.settings), lines_changed_at=now)
            self._states[code] = state
            LOGGER.info("PLATE tracking started camera=%s", code)
        lines = tuple(z for z in config.zones if isinstance(z, CrossingLine))
        if lines != state.lines:
            state.lines = lines
            state.lines_changed_at = now
            for track in state.tracker.tracks.values():
                track.sides.clear()
            LOGGER.info("PLATE crossing lines updated camera=%s lines=%d", code, len(lines))
        for metadata in self.detection.metadata(code):
            if metadata.camera != config.camera or metadata.received_monotonic <= state.last_frame_at:
                continue
            if metadata.received_monotonic < state.lines_changed_at:
                continue
            self._observe(state, metadata)
        while state.history:
            key = next(iter(state.history))
            if len(state.history) <= self.settings.history_frames and now-key[1] <= self.settings.history_seconds:
                break
            state.history.popitem(last=False)
        self._recognize(state)
        state.tracker.expire(now)

    def stop(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(self.detection.settings.shutdown_timeout_seconds)
        stopped = not self._thread.is_alive()
        if stopped:
            if self._pending:
                LOGGER.warning("PLATE passage stopped pending_events=%d result=NOT_PERSISTED", len(self._pending))
            self._pending.clear()
            self._states.clear()
        else:
            LOGGER.error("PLATE passage shutdown exceeded timeout")
        return stopped
