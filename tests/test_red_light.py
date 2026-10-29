import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from plate_ai.overlay.schema import OverlaySnapshot, build_bbox_message
from plate_ai.overlay.config import BboxSettings
from plate_ai.detection.detector import VehicleDetection
from plate_ai.core.domain import Camera, CrossingLine, TrafficZone, ZoneStore
from plate_ai.messaging.events import build_plate_event
from plate_ai.messaging.mqtt import MqttTransport
from plate_ai.tracking.config import PassageSettings
from plate_ai.tracking.runtime import PassageRuntime
from plate_ai.recognition.pipeline import PlateRecognition, VehicleRecognition
from plate_ai.traffic_light.detector import (COLORS, LABELS, LightObservation, RedLightViolationDetector,
                                TrafficLightStateDetector, TrafficLightStateTracker,
                                Violation, draw_traffic_light)
from plate_ai.traffic_light.config import RedLightSettings
from plate_ai.traffic_light.events import PlatePublishGate, RedLightEventPublisher, build_red_light_event
from plate_ai.camera.stream import CapturedFrame
from plate_ai.detection.runtime import DetectionMetadata, DetectionRuntime
from plate_ai.camera.sync import CameraConfiguration
from plate_ai.tracking.tracker import Track

STOP = ((.1, .1), (.9, .1), (.9, .5), (.1, .5))
VIOLATION = ((.1, .5), (.9, .5), (.9, .9), (.1, .9))
LIGHT = ((.01, .01), (.15, .01), (.15, .15), (.01, .15))
CAMERA = Camera("e240a093-fae9-4c25-a11e-1aeb037a0a9b", "CAM_TEST", "rtsp://unused")


def track(y=.4, identity=1):
    return Track(identity, f"actual-track-{identity}", (.4, y-.2, .6, y),
                 VehicleDetection("CAR", .9, (80, 40, 120, 80)), 1.)


def read(t):
    x1, y1, x2, y2 = t.vehicle.bbox
    box = (x1+2, y2-12, x2-2, y2-2)
    return VehicleRecognition(0, t.vehicle, PlateRecognition("30A-12345", .87, box, .9, "WHITE"))


def remember_read(t):
    t.remember(read(t), captured_at=100., sequence=1, identity_epoch=t.identity_epoch)


def flush_ready(publisher, now):
    publisher.flush(now)
    for item in publisher.pending.values():
        if item[6] is not None:
            item[6].result(timeout=3)
    publisher.flush(now)


def signal(state, timestamp):
    return LightObservation(state, .8 if state != "UNKNOWN" else 0., {}, timestamp)


def light_frame(color="RED", shape=(200, 200)):
    frame = np.zeros((*shape, 3), np.uint8)
    cv2.circle(frame, (int(shape[1]*.08), int(shape[0]*.08)), max(2, int(min(shape)*.04)), COLORS[color], -1)
    return frame


def witnessed_violation(*args, **kwargs):
    v = Violation(*args, **kwargs)
    return replace(v, signal=signal("RED", v.frame_timestamp), frame_sequence=1, light_points=LIGHT)


class HsvTests(unittest.TestCase):
    def setUp(self):
        self.settings = RedLightSettings()
        self.detector = TrafficLightStateDetector(self.settings)

    def test_all_colors_with_original_top_of_roi_and_two_red_hues(self):
        for state in ("RED", "GREEN", "YELLOW"):
            with self.subTest(state=state):
                self.assertEqual(self.detector.detect(light_frame(state), LIGHT, 10).state, state)
        for hue in (0, 179):
            hsv = np.zeros((100, 100, 3), np.uint8)
            cv2.circle(hsv, (50, 8), 7, (hue, 255, 255), -1)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.assertEqual(self.detector.detect(frame, ((0, 0), (1, 0), (1, .3), (0, .3)), 1).state, "RED")

    def test_tiny_dark_noise_and_ambiguous_return_unknown(self):
        full = ((0, 0), (1, 0), (1, 1), (0, 1))
        samples = [np.zeros((100, 100, 3), np.uint8), np.full((5, 5, 3), (0, 0, 255), np.uint8)]
        noise = np.zeros((100, 100, 3), np.uint8)
        noise[::3, ::3] = (0, 0, 255)
        samples.append(noise)
        dark = np.zeros_like(noise)
        cv2.circle(dark, (50, 50), 10, (0, 0, 80), -1)
        samples.append(dark)
        ambiguous = np.zeros_like(noise)
        cv2.circle(ambiguous, (25, 50), 10, COLORS["RED"], -1)
        cv2.circle(ambiguous, (75, 50), 10, COLORS["GREEN"], -1)
        samples.append(ambiguous)
        for frame in samples:
            self.assertEqual(self.detector.detect(frame, full, 1).state, "UNKNOWN")

    def test_mask_ignores_color_outside_polygon(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        cv2.circle(frame, (85, 85), 10, COLORS["RED"], -1)
        self.assertEqual(self.detector.detect(frame, ((0, 0), (1, 0), (0, 1)), 1).state, "UNKNOWN")

    def test_fragmented_led_lamps_with_small_area_in_full_housing(self):
        full = ((0, 0), (1, 0), (1, 1), (0, 1))
        for state, hue, brightness, top in (("RED", 177, 210, 80), ("GREEN", 80, 180, 300),
                                            ("YELLOW", 11, 200, 200)):
            hsv = np.zeros((340, 100, 3), np.uint8)
            for y in range(top, top+18, 3):
                for x in range(35, 59, 3):
                    cv2.circle(hsv, (x, y), 1, (hue, 210, brightness), -1)
            original = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            before = original.copy()
            result = self.detector.detect(original, full, 1.)
            with self.subTest(state=state):
                self.assertEqual(result.state, state)
                np.testing.assert_array_equal(original, before)

    def test_amber_hue_is_yellow_and_never_red(self):
        full = ((0, 0), (1, 0), (1, 1), (0, 1))
        for hue in (9, 10, 11, 12, 18, 25, 30):
            hsv = np.zeros((100, 100, 3), np.uint8)
            cv2.circle(hsv, (50, 50), 10, (hue, 200, 200), -1)
            sample = self.detector.detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), full, 1.)
            self.assertEqual(sample.state, "YELLOW", f"hue={hue}")

    def test_causal_smoothing_and_staleness(self):
        tracker = TrafficLightStateTracker(self.settings)
        expected = [("RED", 1., "UNKNOWN"), ("RED", 1.1, "RED"),
                    ("GREEN", 1.2, "UNKNOWN"), ("GREEN", 1.3, "GREEN"),
                    ("RED", 2., "UNKNOWN"), ("RED", 2.1, "RED"),
                    ("UNKNOWN", 2.2, "UNKNOWN"), ("RED", 2.3, "UNKNOWN")]
        for raw, timestamp, result in expected:
            observed = tracker.update(signal(raw, timestamp))
            self.assertEqual((observed.state, observed.timestamp), (result, timestamp))

    def test_preview_labels_colors_copy_and_resize(self):
        for shape in ((200, 400), (400, 200)):
            original = np.zeros((*shape, 3), np.uint8)
            for state, color in COLORS.items():
                with patch("plate_ai.traffic_light.detector.cv2.putText", wraps=cv2.putText) as text:
                    preview = draw_traffic_light(original.copy(), LIGHT, signal(state, 1))
                    self.assertEqual(text.call_args.args[1], LABELS[state])
                    self.assertEqual(text.call_args.args[5], color)
                    x, y = round(.01*(shape[1]-1)), round(.01*(shape[0]-1))
                    np.testing.assert_array_equal(preview[y, x], color)
                    self.assertFalse(original.any())


class SignalHysteresisTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TrafficLightStateTracker(RedLightSettings())

    def update(self, state, timestamp):
        strict = self.tracker.update(signal(state, timestamp))
        return strict.state, self.tracker.display.state

    def establish_red(self):
        for timestamp in (1., 1.1, 1.2):
            self.update("RED", timestamp)

    def test_dropout_and_single_opposing_samples_do_not_flash_preview(self):
        self.establish_red()
        for state, timestamp in (("UNKNOWN", 1.3), ("RED", 1.4),
                                 ("GREEN", 1.5), ("RED", 1.6)):
            strict, display = self.update(state, timestamp)
            self.assertEqual(strict, "UNKNOWN")
            self.assertEqual(display, "RED")
        self.assertEqual(self.update("RED", 1.7), ("RED", "RED"))

    def test_new_phase_requires_consecutive_confirmation_without_intermediate_u(self):
        self.establish_red()
        self.assertEqual(self.update("GREEN", 1.3), ("UNKNOWN", "RED"))
        self.assertEqual(self.update("GREEN", 1.4), ("GREEN", "RED"))
        self.assertEqual(self.update("GREEN", 1.5), ("GREEN", "GREEN"))
        self.assertEqual(self.tracker.display.timestamp, 1.5)

    def test_yellow_also_requires_confirmation(self):
        self.establish_red()
        for timestamp in (1.3, 1.4, 1.5):
            self.update("YELLOW", timestamp)
        self.assertEqual(self.tracker.display.state, "YELLOW")

    def test_unknown_expires_and_held_timestamp_is_not_refreshed(self):
        self.establish_red()
        for timestamp in (1.3, 1.5, 1.7):
            self.assertEqual(self.update("UNKNOWN", timestamp), ("UNKNOWN", "RED"))
            self.assertEqual(self.tracker.display.timestamp, 1.2)
        self.assertEqual(self.update("UNKNOWN", 1.9), ("UNKNOWN", "UNKNOWN"))
        self.assertEqual(self.update("RED", 2.), ("UNKNOWN", "UNKNOWN"))
        self.update("RED", 2.1)
        self.assertEqual(self.update("RED", 2.2), ("RED", "RED"))

    def test_capture_gap_resets_preview_instead_of_retaining_stale_red(self):
        self.establish_red()
        self.assertEqual(self.update("RED", 2.), ("UNKNOWN", "UNKNOWN"))

    def test_reordered_or_duplicate_frame_does_not_advance_confirmation(self):
        self.establish_red()
        saved = self.tracker.display
        for timestamp in (1.2, 1.1):
            self.assertEqual(self.update("GREEN", timestamp), ("UNKNOWN", "RED"))
            self.assertIs(self.tracker.display, saved)
        self.update("GREEN", 1.3)
        self.assertEqual(self.update("GREEN", 1.4)[1], "RED")

    def test_held_red_never_becomes_crossing_evidence(self):
        for crossing_state in ("UNKNOWN", "GREEN", "YELLOW"):
            with self.subTest(crossing_state=crossing_state):
                tracker = TrafficLightStateTracker(RedLightSettings())
                detector = RedLightViolationDetector(STOP, VIOLATION, RedLightSettings())
                events = []
                t = track()
                for i, y in enumerate((.4, .44, .48, .52, .56)):
                    timestamp = 1.+i*.1
                    strict = tracker.update(signal("RED" if i < 3 else crossing_state, timestamp))
                    t.box = (.4, y-.2, .6, y)
                    events.extend(detector.update([t], strict, timestamp, light_frame()))
                    if i == 3:
                        self.assertEqual(tracker.display.state, "RED")
                        self.assertEqual(strict.state, "UNKNOWN")
                self.assertFalse(events)


class TransitionTests(unittest.TestCase):
    def run_path(self, states, positions, *, identities=None, settings=None, violation=VIOLATION):
        detector = RedLightViolationDetector(STOP, violation, settings or RedLightSettings())
        t, events = track(), []
        for i, (state, y) in enumerate(zip(states, positions)):
            if y is None:
                visible = []
            else:
                if identities:
                    t = track(y, identities[i])
                t.box = (.4, y-.2, .6, y)
                visible = [t]
            events.extend(detector.update(visible, signal(state, 100+i*.1), 100+i*.1, light_frame()))
        return events

    def test_red_stop_to_violation_exactly_once(self):
        events = self.run_path(["RED"]*8, [.4, .44, .48, .52, .56, .6, .64, .68])
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].timestamp, 100.25)
        self.assertAlmostEqual(events[0].frame_timestamp, 100.3)

    def test_green_yellow_unknown_no_violation(self):
        for state in ("GREEN", "YELLOW", "UNKNOWN"):
            self.assertFalse(self.run_path([state]*5, [.4, .44, .48, .52, .56]))

    def test_green_crossing_then_red_is_never_reclassified(self):
        self.assertFalse(self.run_path(["GREEN"]*4+["RED"]*4, [.4, .44, .48, .52, .56, .6, .64, .68]))

    def test_red_candidate_keeps_crossing_phase_when_light_changes(self):
        events = self.run_path(["RED"]*4+["GREEN"], [.4, .44, .48, .52, .56])
        self.assertEqual(len(events), 1)

    def test_direct_appearance_reverse_and_boundary_jitter(self):
        for path in ([.54, .58, .6, .64], [.6, .55, .48, .44, .4],
                     [.49, .499, .501, .499, .501, .499, .501]):
            self.assertFalse(self.run_path(["RED"]*len(path), path))

    def test_lost_or_changed_track_id_does_not_establish_crossing(self):
        self.assertFalse(self.run_path(["RED"]*5, [.4, .44, None, .52, .56]))
        self.assertFalse(self.run_path(["RED"]*5, [.4, .44, .48, .52, .56], identities=[1, 1, 1, 2, 2]))

    def test_large_zone_gap_or_time_gap_is_uncertain(self):
        far = ((.1, .6), (.9, .6), (.9, .9), (.1, .9))
        self.assertFalse(self.run_path(["RED"]*5, [.4, .44, .48, .64, .68], violation=far))
        s = replace(RedLightSettings(), max_crossing_seconds=.05)
        self.assertFalse(self.run_path(["RED"]*5, [.4, .44, .48, .52, .56], settings=s))


class ConfigurationTests(unittest.TestCase):
    def test_name_only_schema_line_null_dynamic_invalid_reset(self):
        store = ZoneStore()
        store.apply("PLATE", [{"name": name, "line": None, "zone": points}
                              for name, points in (("TRAFFICLIGHT", LIGHT), ("STOP-ZONE", STOP),
                                                   ("VIOLATION-ZONE", VIOLATION))])
        self.assertEqual(len(store.zones), 3)
        self.assertTrue(all(isinstance(z, TrafficZone) and z.points for z in store.zones))
        store.apply("PLATE/STOP-ZONE", {"name": "STOP-ZONE", "zone": [[0, 0], [1, 1]]})
        self.assertFalse(next(z.points for z in store.zones if z.name == "STOP-ZONE"))
        store.apply("PLATE/STOP-ZONE", None)
        self.assertEqual(len(store.zones), 2)

    def test_self_intersecting_and_pixel_coordinates_rejected(self):
        for points in ([[0, 0], [1, 1], [0, 1], [1, 0]], [[0, 0], [100, 0], [100, 100]]):
            self.assertFalse(TrafficZone.parse({"name": "TRAFFICLIGHT", "zone": points}).points)

    def test_duplicate_named_geometry_replaces_previous_valid_geometry(self):
        store = ZoneStore()
        entry = {"name": "STOP-ZONE", "zone": STOP}
        store.apply("PLATE", [entry])
        self.assertTrue(store.zones[0].points)
        store.apply("PLATE", [entry, entry])
        self.assertFalse(store.zones[0].points)

    def test_settings_validation_and_json_overrides(self):
        s = RedLightSettings.from_env({"RED_LIGHT_ENABLED": "true", "RED_LIGHT_SETTINGS": '{"value_min":180}'})
        self.assertTrue(s.enabled)
        self.assertEqual(s.value_min, 180)
        for values in ({"signal_confirm_frames": 1}, {"dedup_guard_seconds": 1}, {"value_min": 300},
                       {"boundary_margin": float("nan")}, {"frame_buffer": 1.5},
                       {"component_close_size": 4}, {"component_close_size": 17},
                       {"min_component_support": 1.1}, {"preview_confirm_frames": 1},
                       {"preview_confirm_frames": 2.5}, {"preview_hold_seconds": 0}):
            with self.assertRaises(ValueError):
                RedLightSettings(**values)


class DeliveryTests(unittest.TestCase):
    def test_existing_mqtt_transport_qos_ack_disconnect_and_no_duplicate_application_message(self):
        from unittest.mock import Mock
        client = Mock()
        mqtt = SimpleNamespace(MQTT_ERR_SUCCESS=0, MQTT_ERR_NO_CONN=4)
        settings = SimpleNamespace(client_id="test")
        with patch("plate_ai.messaging.mqtt.create_client", return_value=(mqtt, client)):
            transport = MqttTransport(settings)
        t = track()
        remember_read(t)
        event = build_red_light_event(CAMERA, witnessed_violation(t, 100., .8, None), "/actual.png")
        self.assertFalse(transport.publish_plate_event(event, camera_code=CAMERA.code, object_key="red"))
        client.publish.assert_not_called()
        transport._connected.set()
        client.publish.return_value = SimpleNamespace(rc=4, mid=7)
        self.assertTrue(transport.publish_plate_event(event, camera_code=CAMERA.code, object_key="red"))
        self.assertEqual(client.publish.call_count, 1)
        self.assertEqual(client.publish.call_args.args[0], "smart_vms/ai_events/PLATE")
        self.assertEqual(client.publish.call_args.kwargs, {"qos": 1, "retain": False})
        self.assertEqual(transport.publication_status("red")[0], "PENDING")
        with patch("plate_ai.messaging.mqtt.time.monotonic", return_value=22.):
            transport._on_publish(None, None, 7, SimpleNamespace(is_failure=False), None)
        self.assertEqual(transport.publication_status("red"), ("ACK", 22.))
        client.publish.side_effect = RuntimeError("socket unavailable")
        self.assertFalse(transport.publish_plate_event(event, camera_code=CAMERA.code, object_key="next"))

    def test_actual_plate_payload_snapshot_crossing_time_and_no_fake_metadata(self):
        with TemporaryDirectory() as directory:
            t = track()
            remember_read(t)
            v = witnessed_violation(t, 100., .75, light_frame(), 100.1)
            sent = []
            publisher = RedLightEventPublisher(replace(RedLightSettings(), snapshot_dir=directory),
                                              lambda e, **_: sent.append(e) or True)
            try:
                publisher.submit(CAMERA, v)
                item = next(iter(publisher.pending.values()))
                snapshot = item[3].result(timeout=3)
                flush_ready(publisher, item[2]+3)
                self.assertEqual(len(sent), 1)
                event = sent[0]
                self.assertEqual(event["payload"]["violation_type_code"], "RED_LIGHT")
                self.assertEqual(event["payload"]["vehicle"]["license_plate"]["text"], "30A12345")
                self.assertEqual(event["camera_id"], CAMERA.id)
                self.assertEqual(event["payload"]["violation_evidence"]["track_id"], t.id)
                self.assertIsNone(event["snapshot_url"])  # Local paths cannot be resolved by VMS Storage.
                self.assertTrue(Path(snapshot.original).is_file())
                np.testing.assert_array_equal(cv2.imread(snapshot.original), v.frame)
                self.assertNotIn("owner_name", event["payload"]["vehicle"])
                self.assertNotIn("manufacturer", event["payload"]["vehicle"])
                self.assertTrue(event["event_time"].endswith("+07:00"))
                json.dumps(event, allow_nan=False)
            finally:
                publisher.stop()

    def test_delayed_ocr_after_normal_event_is_still_available(self):
        t = track()
        t.crossed_epoch, t.direction = 99, "IN"
        t.event, t.sent = build_plate_event(CAMERA, t), True
        frozen = json.dumps(t.event)
        remember_read(t)
        event = build_red_light_event(CAMERA, witnessed_violation(t, 100, .8, None), "/actual.png")
        self.assertEqual(event["payload"]["vehicle"]["license_plate"]["text"], "30A12345")
        self.assertEqual(json.dumps(t.event), frozen)

    def test_unknown_plate_not_published_disconnect_retry_and_expiry_logged(self):
        with TemporaryDirectory() as directory:
            t, sent = track(), []
            online = [False]
            def publish(e, **kwargs):
                if not online[0]:
                    raise ConnectionError("offline")
                sent.append(e)
                return True
            publisher = RedLightEventPublisher(replace(RedLightSettings(), snapshot_dir=directory), publish)
            try:
                publisher.submit(CAMERA, witnessed_violation(t, 100., .8, light_frame()))
                item = next(iter(publisher.pending.values()))
                item[3].result(timeout=3)
                flush_ready(publisher, item[2]+3)
                self.assertFalse(sent)
                remember_read(t)
                with self.assertLogs("plate_ai.traffic_light.events", level="ERROR"):
                    flush_ready(publisher, item[2]+4)
                self.assertEqual(len(publisher.pending), 1)
                online[0] = True
                publisher.flush(item[2]+6)
                publisher.flush(item[2]+7)
                self.assertEqual(len(sent), 1)
                publisher.submit(CAMERA, witnessed_violation(track(identity=2), 101., .8, light_frame()))
                item = next(iter(publisher.pending.values()))
                with self.assertLogs("plate_ai.traffic_light.events", level="ERROR"):
                    publisher.flush(item[2]+31)
            finally:
                publisher.stop()

    def test_same_plate_dedup_window_waits_for_puback_then_guard(self):
        t = track()
        remember_read(t)
        t.crossed_epoch, t.direction = 100., "IN"
        recognition = build_plate_event(CAMERA, t)
        violation = build_red_light_event(CAMERA, witnessed_violation(t, 100.1, .8, None), "/actual.png")
        delivered, backend_last = [], {}
        clock = [10.]
        def backend(event, **kwargs):
            identity = (event["camera_id"], event["payload"]["vehicle"]["license_plate"]["text"])
            if clock[0]-backend_last.get(identity, -100) > 1:
                delivered.append(event)
                backend_last[identity] = clock[0]
            return True
        receipt = [("PENDING", 10.)]
        gate = PlatePublishGate(backend, 1.25, lambda _: receipt[0])
        with patch("plate_ai.traffic_light.events.time.monotonic", side_effect=lambda: clock[0]):
            self.assertTrue(gate(recognition, camera_code=CAMERA.code, object_key="normal"))
            clock[0] = 10.1
            self.assertFalse(gate(violation, camera_code=CAMERA.code, object_key="red"))
            clock[0] = 15.  # Long disconnect must not release a queued second message.
            self.assertFalse(gate(violation, camera_code=CAMERA.code, object_key="red"))
            receipt[0] = ("ACK", 15.)
            clock[0] = 16.
            self.assertFalse(gate(violation, camera_code=CAMERA.code, object_key="red"))
            clock[0] = 16.3
            self.assertTrue(gate(violation, camera_code=CAMERA.code, object_key="red"))
        self.assertEqual(len(delivered), 2)
        self.assertEqual(delivered[-1]["payload"]["violation_type_code"], "RED_LIGHT")


class PipelineTests(unittest.TestCase):
    def make_runtime(self, enabled=True, directory="unused", publish=None):
        zones = tuple(TrafficZone(name, name, points) for name, points in
                      (("TRAFFICLIGHT", LIGHT), ("STOP-ZONE", STOP), ("VIOLATION-ZONE", VIOLATION)))
        line = CrossingLine("count", ((.1, .5), (.9, .5)))
        config = CameraConfiguration(CAMERA, (*zones, line))
        sync = SimpleNamespace(snapshot=lambda: SimpleNamespace(cameras=(config,)))
        frames, metadata = {}, []
        detection = SimpleNamespace(metadata=lambda _: metadata,
                                    frame_for=lambda m: frames.get(m.sequence),
                                    settings=SimpleNamespace(max_frame_age=1., shutdown_timeout_seconds=1.))
        recognition = SimpleNamespace(latest=lambda _: None)
        runtime = PassageRuntime(sync, detection, recognition, PassageSettings(),
                                 publish=publish if publish is not None else lambda *a, **k: True,
                                 red_light_settings=replace(RedLightSettings(), enabled=enabled, snapshot_dir=directory))
        return runtime, config, frames, metadata

    def step(self, runtime, config, frames, metadata, seq, y, state="RED"):
        now = 100+seq*.1
        frame = light_frame(state)
        vehicle = VehicleDetection("CAR", .9, (80, int((y-.2)*200), 120, int(y*200)))
        m = DetectionMetadata(CAMERA, seq, 1000+seq*.1, now, 200, 200, (vehicle,))
        frames[seq] = CapturedFrame(seq, m.captured_at, now, frame)
        metadata[:] = [m]
        with patch("plate_ai.tracking.runtime.time.monotonic", return_value=now):
            runtime._process_camera(CAMERA.code, config)
        return frame

    def test_shared_tracker_counting_unchanged_with_red_light_and_preview(self):
        with TemporaryDirectory() as directory:
            counts = []
            for enabled in (False, True):
                runtime, config, frames, metadata = self.make_runtime(enabled, directory)
                try:
                    for seq, y in enumerate((.4, .44, .48, .52, .56, .6), 1):
                        original = self.step(runtime, config, frames, metadata, seq, y)
                    state = runtime._states[CAMERA.code]
                    counts.append((state.count_in, state.count_out))
                    if enabled:
                        self.assertEqual(len(runtime.red_events.pending), 1)
                        pending = next(iter(runtime.red_events.pending.values()))
                        pending[3].result(timeout=3)
                        self.assertIs(pending[1].track, next(iter(state.tracker.tracks.values())))
                        with patch("plate_ai.tracking.runtime.time.monotonic", return_value=100.6):
                            preview = runtime.latest_preview(CAMERA.code)
                        self.assertIsNotNone(preview)
                        self.assertFalse(np.array_equal(preview, original))
                        self.assertEqual(state.light.state, "RED")
                finally:
                    runtime.stop()
            self.assertEqual(counts, [(1, 0), (1, 0)])

    def test_live_config_reset_invalid_zones_and_camera_isolation(self):
        runtime, config, frames, metadata = self.make_runtime()
        try:
            for seq, y in enumerate((.4, .44, .48), 1):
                self.step(runtime, config, frames, metadata, seq, y)
            state = runtime._states[CAMERA.code]
            self.assertTrue(state.red_detector.states)
            new_stop = replace(config.zones[1], points=())
            changed = replace(config, zones=(config.zones[0], new_stop, *config.zones[2:]))
            with self.assertLogs("plate_ai.tracking.runtime", level="WARNING"):
                self.step(runtime, changed, frames, metadata, 4, .52)
            self.assertIsNone(state.red_detector)
            self.step(runtime, changed, frames, metadata, 5, .56)
            self.assertFalse(runtime.red_events.pending)
            other = replace(CAMERA, code="CAM_OTHER", id="another-actual-camera")
            with patch("plate_ai.tracking.runtime.time.monotonic", return_value=101.):
                runtime._process_camera(other.code, replace(config, camera=other))
            self.assertIsNot(runtime._states[other.code].light_tracker, state.light_tracker)
        finally:
            runtime.stop()

    def test_missing_evidence_frame_invalidates_crossing(self):
        runtime, config, frames, metadata = self.make_runtime()
        try:
            self.step(runtime, config, frames, metadata, 1, .4)
            state = runtime._states[CAMERA.code]
            frames.clear()
            runtime._observe_red(state, metadata[0])
            self.assertFalse(state.red_detector.states)
            self.assertIsNone(state.preview)
        finally:
            runtime.stop()

    def test_runtime_overlay_holds_red_but_crossing_uses_current_strict_signal(self):
        runtime, config, frames, metadata = self.make_runtime()
        try:
            for seq, y in enumerate((.4, .44, .48), 1):
                self.step(runtime, config, frames, metadata, seq, y)
            self.step(runtime, config, frames, metadata, 4, .52, "GREEN")
            self.step(runtime, config, frames, metadata, 5, .56, "GREEN")
            state = runtime._states[CAMERA.code]
            self.assertEqual(state.light.state, "RED")
            self.assertEqual(state.light_tracker.last.state, "GREEN")
            self.assertFalse(runtime.red_events.pending)
            self.step(runtime, config, frames, metadata, 6, .6, "GREEN")
            self.assertEqual(state.light.state, "GREEN")
            self.assertFalse(runtime.red_events.pending)
        finally:
            runtime.stop()

    def test_actual_crossing_and_ocr_publish_one_combined_event_through_outbox(self):
        from plate_ai.messaging.delivery import EventDelivery, EventDeliverySettings
        with TemporaryDirectory() as directory:
            sent = []
            settings = EventDeliverySettings(state_path=str(Path(directory)/"events.sqlite3"),
                                             recognition_hold_seconds=.5)
            with EventDelivery(settings, lambda e, **_: sent.append(e) or True) as delivery:
                runtime, config, frames, metadata = self.make_runtime(directory=directory, publish=delivery)
                try:
                    self.assertIs(runtime.publish, delivery)
                    self.assertIs(runtime.red_events.publish, delivery)
                    for seq, y in enumerate((.4, .44, .48, .52, .56, .6), 1):
                        self.step(runtime, config, frames, metadata, seq, y)
                    state = runtime._states[CAMERA.code]
                    t = next(iter(state.tracker.tracks.values()))
                    remember_read(t)
                    started = time.monotonic()
                    while runtime._pending or runtime.red_events.pending or not sent:
                        elapsed = time.monotonic()-started
                        self.assertLess(elapsed, 4.)
                        runtime._flush(103.+elapsed)
                        runtime.red_events.flush(103.+elapsed)
                        time.sleep(.01)
                    self.assertEqual(len(sent), 1)
                    self.assertEqual(sent[0]["payload"]["violation_type_code"], "RED_LIGHT")
                    self.assertEqual(sent[0]["payload"]["vehicle"]["license_plate"]["text"], "30A12345")
                    self.assertTrue(sent[0]["event_time"].endswith("+07:00"))
                    self.assertEqual((state.count_in, state.count_out), (1, 0))
                finally:
                    runtime.stop()

    def test_mqtt_overlay_traffic_colors_and_clear(self):
        for state in COLORS:
            snapshot = OverlaySnapshot(CAMERA, 1, 100., time.monotonic(), (), 1, 2, LIGHT, signal(state, 100))
            message = build_bbox_message(snapshot, 1, BboxSettings())
            light = message["detections"][0]
            self.assertEqual(light["text"], LABELS[state])
            self.assertEqual(light["class"], LABELS[state])
            self.assertEqual(light["traffic_light_state"], state)
            b, g, r = COLORS[state]
            self.assertEqual(light["color"], f"#{r:02x}{g:02x}{b:02x}")
            self.assertEqual(message["hud"]["text"], "IN: 1 | OUT: 2")
            self.assertFalse(build_bbox_message(snapshot, 1, BboxSettings(), clear=True)["detections"])


if __name__ == "__main__":
    unittest.main()
