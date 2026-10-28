"""Track camera passages, associate asynchronous OCR and queue MQTT events."""
from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import threading
import time
from .domain import Camera, CrossingLine, TrafficZone
from .events import build_plate_event
from .tracking import VehicleTracker, foot, intersects
from .bbox import OverlayDetection, OverlaySnapshot

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
    count_in: int = 0
    count_out: int = 0
    latest_metadata: object = None
    visible: dict = field(default_factory=dict)
    overlay_key: tuple | None = None
    red_zones: tuple | None = None
    red_changed_at: float = 0.
    red_detector: object = None
    light_tracker: object = None
    light: object = None
    light_points: tuple = ()
    preview: object = None


class PassageRuntime:
    """One metadata controller; inference and network acknowledgements run separately."""
    def __init__(self, sync_service, detection_runtime, recognition_runtime, settings, *, publish=None,
                 emit_events=True, red_light_settings=None):
        self.sync, self.detection, self.recognition = sync_service, detection_runtime, recognition_runtime
        self.settings = settings
        self.emit_events = emit_events
        self.publish = publish if publish is not None else sync_service.transport.publish_plate_event
        self.red_settings = red_light_settings
        self.red_events = None
        if red_light_settings is not None and red_light_settings.enabled:
            from .red_light import TrafficLightStateDetector
            from .red_light_events import PlatePublishGate, RedLightEventPublisher
            status = getattr(getattr(sync_service, "transport", None), "publication_status", None) if publish is None else None
            self.publish = PlatePublishGate(self.publish, red_light_settings.dedup_guard_seconds, status)
            self.red_events = RedLightEventPublisher(red_light_settings, self.publish)
            self.light_detector = TrafficLightStateDetector(red_light_settings)
        self._states, self._pending = {}, {}
        self._overlay_lock = threading.Lock()
        self._overlays = {}
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
        LOGGER.info("PLATE passage worker started events_enabled=%s ocr_wait_seconds=%.1f max_pending=%d",
                    self.emit_events, self.settings.ocr_wait_seconds, self.settings.max_pending)
        return self

    def latest_overlay(self, camera_code):
        with self._overlay_lock:
            snapshot = self._overlays.get(camera_code)
        if snapshot is None or self._stop.is_set():
            return None
        current = next((c.camera for c in self.sync.snapshot().cameras if c.camera.code == camera_code), None)
        return snapshot if current == snapshot.camera else None

    def __enter__(self):
        return self.start()

    def latest_preview(self, camera_code):
        """Native preview frame with vehicle/count/traffic annotations; original stays untouched."""
        snapshot = self.latest_overlay(camera_code)
        with self._overlay_lock:
            state = self._states.get(camera_code)
            if (snapshot is None or state is None or state.preview is None
                    or time.monotonic()-snapshot.received_monotonic > self.detection.settings.max_frame_age):
                return None
            return state.preview.copy()

    def __exit__(self, *_):
        self.stop()

    def _observe(self, state, metadata):
        visible = state.tracker.update(metadata)
        key = (metadata.sequence, metadata.received_monotonic)
        state.history[key] = visible
        state.latest_metadata, state.visible = metadata, visible
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
                    if track.direction == "IN":
                        state.count_in += 1
                    else:
                        state.count_out += 1
                    if self.emit_events and len(self._pending) < self.settings.max_pending:
                        self._pending[track.object_key] = (state.camera, track)
                        LOGGER.info("PLATE crossing camera=%s object_key=%s line=%s direction=%s result=PENDING_OCR",
                                    state.camera.code, track.object_key, line.id, track.direction)
                    elif self.emit_events:
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
                track.remember(read, result.received_monotonic)

    def _flush(self, now):
        for key, (camera, track) in tuple(self._pending.items()):
            current = next((c.camera for c in self.sync.snapshot().cameras if c.camera.code == camera.code), None)
            if current != camera:
                LOGGER.info("PLATE event camera=%s object_key=%s result=CANCELLED_CAMERA_REMOVED", camera.code, key)
                del self._pending[key]
                continue
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
                        if self.red_events:
                            self.red_events.cancel(code)
                        del self._states[code]
                        with self._overlay_lock:
                            self._overlays.pop(code, None)
                        LOGGER.info("PLATE tracking removed camera=%s", code)
                for code, config in configs.items():
                    try:
                        self._process_camera(code, config)
                    except Exception as error:
                        LOGGER.error("PLATE tracking failed camera=%s type=%s", code, type(error).__name__)
                self._flush(time.monotonic())
                if self.red_events:
                    self.red_events.flush(time.monotonic())
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
        if self.red_events:
            self._configure_red(state, config.zones, now)
        for metadata in self.detection.metadata(code):
            if metadata.camera != config.camera or metadata.received_monotonic <= state.last_frame_at:
                continue
            if metadata.received_monotonic < state.lines_changed_at:
                continue
            self._observe(state, metadata)
            if self.red_events:
                try:
                    self._observe_red(state, metadata)
                except Exception as error:
                    if state.red_detector:
                        state.red_detector.states.clear()
                    state.preview, state.light = None, None
                    LOGGER.error("RED_LIGHT camera=%s result=FRAME_REJECTED type=%s", code, type(error).__name__)
        while state.history:
            key = next(iter(state.history))
            if len(state.history) <= self.settings.history_frames and now-key[1] <= self.settings.history_seconds:
                break
            state.history.popitem(last=False)
        self._recognize(state)
        state.tracker.expire(now)
        metadata = state.latest_metadata
        if metadata is not None:
            key = (metadata.sequence, metadata.received_monotonic, state.last_ocr_key)
            if key == state.overlay_key:
                return
            state.overlay_key = key
            detections = tuple(OverlayDetection(t.object_key, t.id, t.vehicle.vehicle_type, t.vehicle.confidence,
                                                t.box, t.plate_relative, t.plate_seen_at, t.plate_text,
                                                t.plate_confidence, t.text_confidence)
                               for t in state.visible.values())
            snapshot = OverlaySnapshot(state.camera, metadata.sequence, metadata.captured_at,
                                       metadata.received_monotonic, detections, state.count_in, state.count_out,
                                       state.light_points, state.light)
            with self._overlay_lock:
                self._overlays[code] = snapshot

    def _configure_red(self, state, zones, now):
        from .red_light import RedLightViolationDetector, TrafficLightStateTracker
        named = tuple(z for z in zones if isinstance(z, TrafficZone))
        if named == state.red_zones:
            return
        state.red_zones, state.red_changed_at = named, now
        state.red_detector, state.light, state.preview = None, None, None
        state.light_points = ()
        state.overlay_key = None
        self.red_events.cancel(state.camera.code)
        by_name = {}
        for zone in named:
            by_name.setdefault(zone.name, []).append(zone)
        light = by_name.get("TRAFFICLIGHT", ())
        if len(light) == 1 and light[0].points:
            state.light_points = light[0].points
        valid = all(len(by_name.get(name, ())) == 1 and by_name[name][0].points
                    for name in ("TRAFFICLIGHT", "STOP-ZONE", "VIOLATION-ZONE"))
        state.light_tracker = TrafficLightStateTracker(self.red_settings)
        if valid:
            state.red_detector = RedLightViolationDetector(by_name["STOP-ZONE"][0].points,
                                                         by_name["VIOLATION-ZONE"][0].points, self.red_settings)
        LOGGER.log(logging.INFO if valid else logging.WARNING,
                   "RED_LIGHT camera=%s result=%s", state.camera.code,
                   "CONFIG_ENABLED" if valid else "DISABLED_MISSING_OR_INVALID_ZONES")

    def _observe_red(self, state, metadata):
        from .red_light import LightObservation, draw_traffic_light
        import cv2
        if metadata.received_monotonic < state.red_changed_at:
            return
        captured = self.detection.frame_for(metadata)
        if captured is None:
            if state.red_detector:
                state.red_detector.states.clear()
            state.light, state.preview = None, None
            return
        frame = captured.image
        if state.light_points:
            raw = self.light_detector.detect(frame, state.light_points, metadata.captured_at)
            state.light = state.light_tracker.update(raw)
        else:
            state.light = LightObservation("UNKNOWN", 0., {}, metadata.captured_at)
        if state.red_detector:
            if time.monotonic()-metadata.received_monotonic <= self.red_settings.signal_max_age:
                events = state.red_detector.update(list(state.visible.values()), state.light,
                                                   metadata.captured_at, frame)
                for event in events:
                    self.red_events.submit(state.camera, event)
            else:
                state.red_detector.states.clear()
        preview = frame.copy()
        h, w = preview.shape[:2]
        for track in state.visible.values():
            x1, y1, x2, y2 = (int(v*(w if i % 2 == 0 else h)) for i, v in enumerate(track.box))
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(preview, f"#{track.id} {track.plate_text}", (max(0, x1), max(15, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 180, 0), 1)
        cv2.putText(preview, f"IN: {state.count_in} | OUT: {state.count_out}", (4, max(15, h-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        if state.light_points:
            draw_traffic_light(preview, state.light_points, state.light, debug=self.red_settings.debug)
        with self._overlay_lock:
            state.preview = preview

    def stop(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(self.detection.settings.shutdown_timeout_seconds)
        stopped = not self._thread.is_alive()
        if stopped:
            if self.red_events:
                self.red_events.stop()
            if self._pending:
                LOGGER.warning("PLATE passage stopped pending_events=%d result=NOT_PERSISTED", len(self._pending))
            self._pending.clear()
            self._states.clear()
            with self._overlay_lock:
                self._overlays.clear()
        else:
            LOGGER.error("PLATE passage shutdown exceeded timeout")
        return stopped
