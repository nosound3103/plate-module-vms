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
    history_identities: dict = field(default_factory=dict)
    last_ocr_key: tuple | None = None
    ocr_revision: int = 0
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
                 emit_events=True, red_light_settings=None, snapshots=None, red_light_snapshots=None):
        self.sync, self.detection, self.recognition = sync_service, detection_runtime, recognition_runtime
        self.settings = settings
        self.emit_events = emit_events
        self.snapshots = snapshots
        self._snapshot_futures = {}
        self.publish = publish if publish is not None else sync_service.transport.publish_plate_event
        self.red_settings = red_light_settings
        self.red_events = None
        if red_light_settings is not None and red_light_settings.enabled:
            from .red_light import TrafficLightStateDetector
            from .red_light_events import PlatePublishGate, RedLightEventPublisher
            status = getattr(getattr(sync_service, "transport", None), "publication_status", None) if publish is None else None
            if not getattr(self.publish, "manages_delivery", False):
                self.publish = PlatePublishGate(self.publish, red_light_settings.dedup_guard_seconds, status)
            self.red_events = RedLightEventPublisher(red_light_settings, self.publish,
                                                    snapshots=red_light_snapshots if red_light_snapshots is not None else snapshots)
            self.light_detector = TrafficLightStateDetector(red_light_settings)
        self._states, self._pending = {}, {}
        self._overlay_lock = threading.Lock()
        self._overlays = {}
        self._recognition_targets, self._subscribers = {}, set()
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, name="plate-passages", daemon=True)
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("This passage runtime has already been started")
        if not self.detection.metadata_enabled:
            raise ValueError("Passage runtime requires detection metadata history")
        self._started = True
        if hasattr(self.detection, "subscribe_updates"):
            self.detection.subscribe_updates(self._wake)
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

    def subscribe_updates(self, event):
        with self._overlay_lock:
            self._subscribers.add(event)
        event.set()

    def unsubscribe_updates(self, event):
        with self._overlay_lock:
            self._subscribers.discard(event)

    def wake(self):
        self._wake.set()

    def recognition_targets(self, detected):
        """Immutable indices/track identities; exposed only after the fast overlay."""
        with self._overlay_lock:
            entry = self._recognition_targets.get(detected.camera.code)
        if entry is None:
            return None
        metadata, targets, active = entry
        if (metadata.camera != detected.camera or metadata.sequence != detected.frame.sequence
                or metadata.received_monotonic != detected.frame.received_monotonic):
            return None
        return targets, active

    def __enter__(self):
        return self.start()

    def latest_preview(self, camera_code):
        """Native preview frame with vehicle/count/traffic annotations; original stays untouched."""
        import cv2
        from .red_light import draw_traffic_light
        snapshot = self.latest_overlay(camera_code)
        with self._overlay_lock:
            state = self._states.get(camera_code)
            if (snapshot is None or state is None or state.preview is None
                    or time.monotonic()-snapshot.received_monotonic > self.detection.settings.max_frame_age):
                return None
            source, vehicles, count_in, count_out, light_points, light = state.preview
        # Preview drawing is optional and never holds the bbox snapshot lock.
        preview = source.copy()
        h, w = preview.shape[:2]
        for box, identity, plate in vehicles:
            x1, y1, x2, y2 = (int(v*(w if i % 2 == 0 else h)) for i, v in enumerate(box))
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(preview, f"#{identity} {plate}", (max(0, x1), max(15, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 180, 0), 1)
        cv2.putText(preview, f"IN: {count_in} | OUT: {count_out}", (4, max(15, h-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        if light_points:
            draw_traffic_light(preview, light_points, light, debug=self.red_settings.debug)
        return preview

    def __exit__(self, *_):
        self.stop()

    def _observe(self, state, metadata):
        visible = state.tracker.update(metadata)
        key = (metadata.sequence, metadata.received_monotonic)
        state.history[key] = visible
        state.history_identities[key] = {i: (t.object_key, t.identity_epoch, t.vehicle.bbox, t.association_reliable)
                                         for i, t in visible.items()}
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
                        if self.snapshots:
                            from .passage_evidence import PassageAnnotation
                            captured = self.detection.frame_for(metadata)
                            self._snapshot_futures[track.object_key] = self.snapshots.submit(
                                state.camera, captured.image if captured else None, object_key=track.object_key,
                                annotation=PassageAnnotation(tuple(track.box), track.id, track.direction))
                        LOGGER.info("PLATE crossing camera=%s object_key=%s line=%s direction=%s result=PENDING_OCR",
                                    state.camera.code, track.object_key, line.id, track.direction)
                    elif self.emit_events:
                        LOGGER.error("PLATE crossing camera=%s object_key=%s result=DROPPED_PENDING_FULL",
                                     state.camera.code, track.object_key)
                track.sides[line.id] = (side, point)

    def _recognize(self, state):
        if self.recognition is None:
            return
        if hasattr(self.recognition, "drain"):
            results = self.recognition.drain(state.camera.code)
        else:
            result = self.recognition.latest(state.camera.code)
            results = (result,) if result is not None else ()
        for result in results:
            if result.camera != state.camera:
                continue
            key = (result.sequence, result.received_monotonic)
            if not hasattr(self.recognition, "drain") and key == state.last_ocr_key:
                continue
            associations = state.history.get(key, {})
            for read in result.vehicles:
                track = associations.get(read.detection_index)
                if read.object_key is not None:
                    if track is None:
                        track = next((t for t in state.tracker.tracks.values() if t.object_key == read.object_key), None)
                    if track is None:
                        pending = self._pending.get(read.object_key)
                        if pending is not None and pending[0] == state.camera:
                            track = pending[1]
                    if track is None and self.red_events:
                        track = next((item[1].track for item in self.red_events.pending.values()
                                      if item[0] == state.camera and item[1].track.object_key == read.object_key), None)
                    if track is None or track.object_key != read.object_key:
                        continue
                if track is not None:
                    proof = state.history_identities.get(key, {}).get(read.detection_index)
                    if proof and (proof[0] != track.object_key or proof[1] != track.identity_epoch
                                  or tuple(proof[2]) != tuple(read.vehicle.bbox)):
                        LOGGER.warning("PLATE OCR camera=%s sequence=%s result=REJECTED_IDENTITY_MISMATCH",
                                       state.camera.code, result.sequence)
                        continue
                    track.remember(read, result.received_monotonic,
                                   captured_at=result.captured_at if proof and proof[3] else None,
                                   sequence=result.sequence, identity_epoch=proof[1] if proof else None)
            # Each drained result can contain one track from the same frame.
            state.last_ocr_key = key
            state.ocr_revision += 1

    def _flush(self, now):
        for key, (camera, track) in tuple(self._pending.items()):
            current = next((c.camera for c in self.sync.snapshot().cameras if c.camera.code == camera.code), None)
            if current != camera:
                LOGGER.info("PLATE event camera=%s object_key=%s result=CANCELLED_CAMERA_REMOVED", camera.code, key)
                del self._pending[key]
                self._cancel_snapshot(key)
                continue
            age = now-track.crossed_monotonic
            if age > self.settings.pending_ttl:
                LOGGER.error("PLATE event camera=%s object_key=%s result=DROPPED_EXPIRED", camera.code, key)
                del self._pending[key]
                self._cancel_snapshot(key)
                continue
            if age < self.settings.ocr_wait_seconds or now < track.retry_at:
                continue
            if track.event is None:
                snapshot = None
                if key in self._snapshot_futures:
                    ready, snapshot = self.snapshots.resolve(self._snapshot_futures[key], age)
                    if not ready:
                        continue
                track.event = build_plate_event(camera, track)
                track.event["snapshot_url"] = snapshot
                self._snapshot_futures.pop(key, None)
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

    def _cancel_snapshot(self, key):
        future = self._snapshot_futures.pop(key, None)
        if future:
            future.cancel()

    def _run(self):
        while not self._stop.is_set():
            self._wake.clear()
            try:
                configs = {c.camera.code: c for c in self.sync.snapshot().cameras}
                for code, state in tuple(self._states.items()):
                    if code not in configs or configs[code].camera != state.camera:
                        if self.red_events:
                            self.red_events.cancel(code)
                        del self._states[code]
                        with self._overlay_lock:
                            self._overlays.pop(code, None)
                            self._recognition_targets.pop(code, None)
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
            self._wake.wait(.02)

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
            # Publish tracked geometry before HSV, preview copies or OCR work.
            self._publish_overlay(state)
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
            expired, _ = state.history.popitem(last=False)
            state.history_identities.pop(expired, None)
        self._recognize(state)
        state.tracker.expire(now)
        self._publish_overlay(state)

    def _publish_overlay(self, state):
        metadata = state.latest_metadata
        if metadata is not None:
            key = (metadata.sequence, metadata.received_monotonic, state.ocr_revision, state.light)
            if key == state.overlay_key:
                return
            state.overlay_key = key
            detections = tuple(OverlayDetection(t.object_key, t.id, t.vehicle.vehicle_type, t.vehicle.confidence,
                                                t.box, t.plate_relative, t.plate_seen_at, t.plate_text,
                                                t.plate_confidence, t.text_confidence)
                               for t in state.visible.values())
            snapshot = OverlaySnapshot(state.camera, metadata.sequence, metadata.captured_at,
                                       metadata.received_monotonic, detections, state.count_in, state.count_out,
                                       state.light_points, state.light, metadata.completed_monotonic)
            targets = tuple((i, t.object_key) for i, t in state.visible.items() if t.best_read() is None)
            active = frozenset(t.object_key for t in state.tracker.tracks.values())
            with self._overlay_lock:
                self._overlays[state.camera.code] = snapshot
                self._recognition_targets[state.camera.code] = (metadata, targets, active)
                subscribers = tuple(self._subscribers)
            for event in subscribers:
                event.set()

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
                                                         by_name["VIOLATION-ZONE"][0].points, self.red_settings,
                                                         light_points=state.light_points)
        LOGGER.log(logging.INFO if valid else logging.WARNING,
                   "RED_LIGHT camera=%s result=%s", state.camera.code,
                   "CONFIG_ENABLED" if valid else "DISABLED_MISSING_OR_INVALID_ZONES")

    def _observe_red(self, state, metadata):
        from .red_light import LightObservation
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
            crossing_signal = state.light_tracker.update(raw)
            state.light = state.light_tracker.display
        else:
            state.light = LightObservation("UNKNOWN", 0., {}, metadata.captured_at)
            crossing_signal = state.light
        if state.red_detector:
            if time.monotonic()-metadata.received_monotonic <= self.red_settings.signal_max_age:
                events = state.red_detector.update(list(state.visible.values()), crossing_signal,
                                                   metadata.captured_at, frame, frame_sequence=metadata.sequence)
                for event in events:
                    self.red_events.submit(state.camera, event)
            else:
                state.red_detector.states.clear()
        with self._overlay_lock:
            state.preview = (frame, tuple((tuple(t.box), t.id, t.plate_text) for t in state.visible.values()),
                             state.count_in, state.count_out, state.light_points, state.light)

    def stop(self):
        self._stop.set()
        self._wake.set()
        if hasattr(self.detection, "unsubscribe_updates"):
            self.detection.unsubscribe_updates(self._wake)
        if self._thread.ident is not None:
            self._thread.join(self.detection.settings.shutdown_timeout_seconds)
        stopped = not self._thread.is_alive()
        if stopped:
            if self.red_events:
                self.red_events.stop()
            if self._pending:
                LOGGER.warning("PLATE passage stopped pending_events=%d result=NOT_PERSISTED", len(self._pending))
            self._pending.clear()
            for key in tuple(self._snapshot_futures):
                self._cancel_snapshot(key)
            self._states.clear()
            with self._overlay_lock:
                self._overlays.clear()
                self._recognition_targets.clear()
        else:
            LOGGER.error("PLATE passage shutdown exceeded timeout")
        return stopped
