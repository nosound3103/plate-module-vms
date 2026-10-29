from dataclasses import replace
import os
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from plate_ai.bbox import OverlaySnapshot, build_bbox_message
from plate_ai.bbox_config import BboxSettings
from plate_ai.bbox_runtime import BboxRuntime
from plate_ai.detection import VehicleDetection
from plate_ai.domain import Camera
from plate_ai.service import CameraConfiguration
from plate_ai.passage_config import PassageSettings
from plate_ai.passage_runtime import PassageRuntime
from plate_ai.recognition import PlateRecognition, RecognitionPipeline, VehicleRecognition
from plate_ai.recognition_config import RecognitionSettings
from plate_ai.recognition_runtime import RecognitionRuntime
from plate_ai.rtsp import CapturedFrame
from plate_ai.runtime import DetectionMetadata, DetectionResult


CAMERA = Camera("camera-test-uuid", "CAM_TEST", "rtsp://unused")
CAR = VehicleDetection("CAR", .9, (10, 10, 40, 60))


def wait_for(predicate, timeout=3.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Pipeline condition was not reached")


class Sync:
    settings = SimpleNamespace(organization_id=1)

    def __init__(self, cameras=(CAMERA,)):
        self.configs = tuple(CameraConfiguration(c, ()) for c in cameras)

    def snapshot(self):
        return SimpleNamespace(cameras=self.configs)


class Detection:
    metadata_enabled = True
    settings = SimpleNamespace(shutdown_timeout_seconds=2., max_frame_age=1.)

    def __init__(self):
        self.results, self.histories, self.subscribers = {}, {}, set()
        self._last_emit = 0.

    def latest(self, code):
        return self.results.get(code)

    def metadata(self, code):
        return tuple(self.histories.get(code, ()))

    def subscribe_updates(self, event):
        self.subscribers.add(event)
        event.set()

    def unsubscribe_updates(self, event):
        self.subscribers.discard(event)

    def emit(self, sequence, vehicles=(CAR,), camera=CAMERA):
        now = max(time.monotonic(), self._last_emit+.001)
        self._last_emit = now
        frame = CapturedFrame(sequence, time.time(), now, np.zeros((100, 100, 3), np.uint8))
        result = DetectionResult(camera, frame, tuple(vehicles), 10., now)
        self.results[camera.code] = result
        self.histories.setdefault(camera.code, []).append(
            DetectionMetadata(camera, sequence, frame.captured_at, now, 100, 100, tuple(vehicles), now))
        for event in tuple(self.subscribers):
            event.set()
        return result


class PriorityTests(unittest.TestCase):
    def test_bbox_advances_while_red_light_snapshot_upload_is_blocked(self):
        from plate_ai.snapshots import SnapshotService
        from plate_ai.snapshot_config import SnapshotSettings
        from plate_ai.red_light import Violation, LightObservation
        from plate_ai.red_light_config import RedLightSettings
        from plate_ai.red_light_events import RedLightEventPublisher
        from plate_ai.tracking import Track
        entered, release = threading.Event(), threading.Event()
        sent, events = [], []
        def upload(camera, image, *, object_key):
            entered.set()
            if not release.wait(10.):
                raise TimeoutError("test upload was not released")
            return "plate/marked.jpg" if object_key.endswith(":annotated") else "plate/original.jpg"
        with TemporaryDirectory() as directory:
            with SnapshotService("unused", replace(SnapshotSettings(), wait_seconds=20.),
                                 client=SimpleNamespace(upload=upload)) as storage:
                self.assertNotEqual(storage.image_executor.submit(os.getpid).result(timeout=3), os.getpid())
                sync, detection = Sync(), Detection()
                passages = PassageRuntime(sync, detection, None, PassageSettings(),
                                          publish=lambda *a, **k: True, emit_events=False)
                publisher = RedLightEventPublisher(replace(RedLightSettings(), snapshot_dir=directory,
                                                            ocr_wait_seconds=.01),
                                                   lambda event, **_: events.append(event) or True, snapshots=storage)
                self.assertIs(publisher.executor, storage.image_executor)
                passages.red_events = publisher
                passages._configure_red = lambda *args: None
                passages._observe_red = lambda *args: None
                boxes = BboxRuntime(sync, passages, BboxSettings(max_fps=200., max_frame_age=1.),
                                    publish=lambda message, **_: sent.append(message) or True)
                passages.start()
                boxes.start()
                try:
                    wait_for(lambda: CAMERA.code in passages._states)
                    for index in range(3):
                        t = Track(index+1, f"violation-track-{index}", (.1, .1, .4, .6), CAR, 100.)
                        t.remember(VehicleRecognition(0, CAR, PlateRecognition(f"30A{12345+index}", .9,
                                                                               (18, 40, 32, 50), .9)),
                                   captured_at=99., sequence=1, identity_epoch=0)
                        publisher.submit(CAMERA, Violation(t, 100., .9, np.zeros((100, 100, 3), np.uint8),
                                         100., 1, LightObservation("RED", .9, {}, 100.),
                                         ((0., 0.), (.2, 0.), (.2, .2), (0., .2))))
                    self.assertTrue(entered.wait(5.))
                    for seq in range(1, 6):
                        detection.emit(seq, (replace(CAR, bbox=(10+seq, 10, 40+seq, 60)),))
                        wait_for(lambda: any(m["frame_sequence"] == seq and m["detections"] for m in sent))
                    self.assertFalse(events)
                    self.assertFalse(release.is_set())
                    release.set()
                    wait_for(lambda: len(events) == 3)
                    self.assertEqual(len({event["entity_id"] for event in events}), 3)
                    self.assertTrue(all(event["snapshot_url"] == "plate/marked.jpg" for event in events))
                finally:
                    release.set()
                    boxes.stop()
                    passages.stop()

    def test_slow_preview_render_does_not_hold_bbox_lock_or_stop_next_frame(self):
        entered, release = threading.Event(), threading.Event()
        sync, detection, sent = Sync(), Detection(), []
        passages = PassageRuntime(sync, detection, None, PassageSettings(),
                                  publish=lambda *a, **k: True, emit_events=False)
        boxes = BboxRuntime(sync, passages, BboxSettings(max_fps=200., max_frame_age=1.),
                            publish=lambda message, **_: sent.append(message) or True)
        passages.start()
        boxes.start()
        preview_thread = None
        try:
            wait_for(lambda: CAMERA.code in passages._states)
            detection.emit(1)
            wait_for(lambda: bool(sent))
            state = passages._states[CAMERA.code]
            state.preview = (np.zeros((100, 100, 3), np.uint8), (((.1, .1, .4, .6), 1, ""),),
                             0, 0, (), None)
            def draw(*args, **kwargs):
                entered.set()
                release.wait(3.)
            with patch("cv2.rectangle", side_effect=draw):
                preview_thread = threading.Thread(target=passages.latest_preview, args=(CAMERA.code,))
                preview_thread.start()
                self.assertTrue(entered.wait(1.))
                detection.emit(2)
                wait_for(lambda: any(m["frame_sequence"] == 2 for m in sent))
                self.assertFalse(release.is_set())
                release.set()
                preview_thread.join(2.)
                self.assertFalse(preview_thread.is_alive())
        finally:
            release.set()
            if preview_thread:
                preview_thread.join(2.)
            boxes.stop()
            passages.stop()

    def test_ocr_enrichment_reserves_capacity_for_next_vehicle_frame(self):
        sync, sent = Sync(), []
        runtime = BboxRuntime(sync, None, BboxSettings(max_fps=10.),
                              publish=lambda message, **_: sent.append(message) or True)
        geometry = OverlaySnapshot(CAMERA, 1, 100., 10., (), 0, 0)
        enriched = replace(geometry, count_in=1)
        next_frame = replace(geometry, sequence=2, received_monotonic=10.05)
        with patch("plate_ai.bbox_runtime.time.monotonic", return_value=10.):
            self.assertTrue(runtime._send_latest(geometry))
        with patch("plate_ai.bbox_runtime.time.monotonic", return_value=10.04):
            self.assertFalse(runtime._send_latest(enriched))
        with patch("plate_ai.bbox_runtime.time.monotonic", return_value=10.05):
            self.assertTrue(runtime._send_latest(next_frame))
        self.assertEqual([m["frame_sequence"] for m in sent], [1, 2])
        with patch("plate_ai.bbox_runtime.time.monotonic", return_value=10.051):
            self.assertFalse(runtime._send_latest(replace(next_frame, count_in=2)))

    def test_boxes_are_available_before_slow_traffic_preview(self):
        entered, release = threading.Event(), threading.Event()
        sync, detection, sent = Sync(), Detection(), []
        passages = PassageRuntime(sync, detection, None, PassageSettings(),
                                  publish=lambda *a, **k: True, emit_events=False)
        passages.red_events = Mock()
        passages._configure_red = lambda *args: None
        def slow_preview(*args):
            entered.set()
            release.wait(2.)
        passages._observe_red = slow_preview
        boxes = BboxRuntime(sync, passages, BboxSettings(max_fps=200., max_frame_age=1.),
                            publish=lambda message, **_: sent.append(message) or True)
        passages.start()
        boxes.start()
        try:
            wait_for(lambda: CAMERA.code in passages._states)
            detection.emit(1)
            self.assertTrue(entered.wait(1.))
            wait_for(lambda: bool(sent))
            self.assertEqual(sent[0]["frame_sequence"], 1)
            self.assertFalse(release.is_set())
        finally:
            release.set()
            boxes.stop()
            passages.stop()

    def test_slow_ocr_does_not_delay_boxes_and_success_is_reused_for_track(self):
        entered, release = threading.Event(), threading.Event()
        calls, sent = [], []

        class Pipeline:
            def process(self, result, zones, *, detection_indices):
                calls.append(result.frame.sequence)
                entered.set()
                release.wait(2.)
                return tuple(VehicleRecognition(i, result.detections[i],
                             PlateRecognition("30A12345", .9, (18, 40, 32, 50), .9))
                             for i in detection_indices)

        sync, detection = Sync(), Detection()
        recognition = RecognitionRuntime(sync, detection, RecognitionSettings(interval_seconds=.01),
                                         pipeline_factory=lambda *_: Pipeline())
        passages = PassageRuntime(sync, detection, recognition, PassageSettings(),
                                  publish=lambda *a, **k: True, emit_events=False)
        recognition.track_source = passages
        boxes = BboxRuntime(sync, passages, BboxSettings(max_fps=200., max_frame_age=1.),
                            publish=lambda message, **_: sent.append(message) or True)
        passages.start()
        boxes.start()
        recognition.start()
        try:
            wait_for(lambda: CAMERA.code in passages._states)
            detection.emit(1)
            self.assertTrue(entered.wait(2.))
            wait_for(lambda: any(m["frame_sequence"] == 1 for m in sent))
            detection.emit(2, (replace(CAR, bbox=(12, 10, 42, 60)),))
            wait_for(lambda: any(m["frame_sequence"] == 2 for m in sent))
            first, second = sent[0], next(m for m in sent if m["frame_sequence"] == 2)
            self.assertEqual(first["detections"][0]["id"], second["detections"][0]["id"])
            self.assertEqual([d["class"] for d in second["detections"]], ["car"])
            self.assertFalse(release.is_set())
            release.set()
            wait_for(lambda: any(d.get("text") == "30A12345" for m in sent for d in m["detections"]))
            for sequence in range(3, 8):
                detection.emit(sequence, (replace(CAR, bbox=(sequence+10, 10, sequence+40, 60)),))
                wait_for(lambda: passages.latest_overlay(CAMERA.code).sequence == sequence)
            time.sleep(.04)
            self.assertEqual(calls, [1])
            overlay = passages.latest_overlay(CAMERA.code)
            with patch("plate_ai.bbox.time.monotonic", return_value=time.monotonic()+10):
                message = build_bbox_message(overlay, 1, BboxSettings(plate_max_age=.01))
            self.assertTrue(any(d.get("text") == "30A12345" for d in message["detections"]))
        finally:
            release.set()
            self.assertTrue(recognition.stop())
            self.assertTrue(boxes.stop())
            self.assertTrue(passages.stop())

    def test_boxes_work_without_ocr_and_reconnect_uses_latest_geometry(self):
        sync, detection, sent, online = Sync(), Detection(), [], [False]
        passages = PassageRuntime(sync, detection, None, PassageSettings(),
                                  publish=lambda *a, **k: True, emit_events=False)
        def publish(message, **kwargs):
            if online[0]:
                sent.append(message)
            return online[0]
        boxes = BboxRuntime(sync, passages, BboxSettings(max_fps=200., max_frame_age=.1), publish=publish)
        passages.start()
        boxes.start()
        try:
            wait_for(lambda: CAMERA.code in passages._states)
            detection.emit(1)
            wait_for(lambda: passages.latest_overlay(CAMERA.code) is not None)
            detection.emit(2)
            wait_for(lambda: passages.latest_overlay(CAMERA.code).sequence == 2)
            online[0] = True
            wait_for(lambda: bool(sent))
            self.assertEqual(sent[0]["frame_sequence"], 2)
            time.sleep(.04)
            self.assertEqual(len(sent), 1)  # No redundant same-frame heartbeat.
            wait_for(lambda: sent[-1]["detections"] == [])
        finally:
            boxes.stop()
            passages.stop()

    def test_delayed_read_uses_original_track_when_detection_order_changes(self):
        sync, detection = Sync(), Detection()
        queued = []
        recognition = SimpleNamespace(drain=lambda _: tuple(queued))
        passages = PassageRuntime(sync, detection, recognition, PassageSettings(),
                                  publish=lambda *a, **k: True, emit_events=False)
        config = sync.configs[0]
        passages._process_camera(CAMERA.code, config)
        other = replace(CAR, bbox=(60, 10, 90, 60))
        original = detection.emit(1, (CAR, other))
        passages._process_camera(CAMERA.code, config)
        first_track = passages._states[CAMERA.code].visible[0]
        detection.emit(2, (other, CAR))
        passages._process_camera(CAMERA.code, config)
        from plate_ai.recognition_runtime import RecognitionResult
        read = VehicleRecognition(0, CAR, PlateRecognition("30A12345", .9), object_key=first_track.object_key)
        queued.append(RecognitionResult(CAMERA, 1, original.frame.captured_at,
                                       original.frame.received_monotonic, (read,), 20.))
        passages._process_camera(CAMERA.code, config)
        visible = passages._states[CAMERA.code].visible
        self.assertEqual(visible[1].plate_text, "30A12345")
        self.assertEqual(visible[0].plate_text, "")
        # A mismatched identity must never attach to a reused detection index.
        queued[:] = [replace(queued[0], vehicles=(replace(read, object_key="another-track"),))]
        passages._process_camera(CAMERA.code, config)
        self.assertEqual(visible[0].plate_text, "")
        # A crossing can still be awaiting OCR after its live track/history expire.
        state = passages._states[CAMERA.code]
        state.tracker.tracks.clear()
        state.history.clear()
        passages._pending[first_track.object_key] = (CAMERA, first_track)
        queued[:] = [replace(queued[0], vehicles=(replace(read, license_plate=PlateRecognition("29B12345", .95)),))]
        passages._recognize(state)
        self.assertEqual(first_track.plate_text, "29B12345")
        passages.stop()


class RecognitionSchedulingTests(unittest.TestCase):
    def make_runtime(self, **settings):
        sync, detection = Sync(), Detection()
        state = [((0, "track-1"),), frozenset({"track-1"})]
        source = SimpleNamespace(recognition_targets=lambda _: tuple(state), wake=lambda: None)
        runtime = RecognitionRuntime(sync, detection, RecognitionSettings(interval_seconds=.001, **settings),
                                     track_source=source)
        runtime.pipeline = Mock()
        return runtime, sync, detection, state

    def process(self, runtime, sync, detection, sequence, vehicles=(CAR,)):
        detection.emit(sequence, vehicles)
        runtime._next_at[CAMERA.code] = 0.
        runtime._process_camera(CAMERA.code, sync.configs[0])

    def test_success_cached_before_handoff_unreadable_retries_and_new_identity_is_read(self):
        runtime, sync, detection, state = self.make_runtime()
        runtime.pipeline.process.side_effect = [
            (VehicleRecognition(0, CAR, PlateRecognition()),),
            (VehicleRecognition(0, CAR, PlateRecognition("30A12345", .9)),),
            (VehicleRecognition(0, CAR, PlateRecognition("29B12345", .9)),)]
        self.process(runtime, sync, detection, 1)
        self.process(runtime, sync, detection, 2)
        self.process(runtime, sync, detection, 3)
        self.assertEqual(runtime.pipeline.process.call_count, 2)
        self.assertEqual(runtime.drain(CAMERA.code)[0].vehicles[0].license_plate.text, "30A12345")
        state[:] = [((0, "track-2"),), frozenset({"track-2"})]
        self.process(runtime, sync, detection, 4)
        self.assertEqual(runtime.pipeline.process.call_count, 3)
        self.assertEqual(runtime.drain(CAMERA.code)[0].vehicles[0].object_key, "track-2")
        self.assertEqual(runtime._known[CAMERA.code], {"track-2"})

    def test_backpressure_preserves_completed_reads_and_fairly_rotates_unreadable_tracks(self):
        runtime, sync, detection, state = self.make_runtime(max_vehicles=1, max_pending_results=1)
        state[:] = [((0, "track-1"), (1, "track-2")), frozenset({"track-1", "track-2"})]
        runtime.pipeline.process.side_effect = lambda r, z, detection_indices: tuple(
            VehicleRecognition(i, r.detections[i], PlateRecognition()) for i in detection_indices)
        self.process(runtime, sync, detection, 1, (CAR, CAR))
        self.process(runtime, sync, detection, 2, (CAR, CAR))
        self.assertEqual(runtime.pipeline.process.call_count, 1)
        self.assertEqual(runtime.drain(CAMERA.code)[0].vehicles[0].object_key, "track-1")
        self.process(runtime, sync, detection, 3, (CAR, CAR))
        self.assertEqual(runtime.drain(CAMERA.code)[0].vehicles[0].object_key, "track-2")

    def test_configuration_change_during_ocr_discards_result(self):
        runtime, sync, detection, _ = self.make_runtime()
        def process(*args, **kwargs):
            sync.configs = (replace(sync.configs[0], camera=replace(CAMERA, source="rtsp://unused/changed")),)
            return (VehicleRecognition(0, CAR, PlateRecognition("30A12345", .9)),)
        runtime.pipeline.process.side_effect = process
        self.process(runtime, sync, detection, 1)
        self.assertIsNone(runtime.latest(CAMERA.code))
        self.assertFalse(runtime.drain(CAMERA.code))
        self.assertFalse(runtime._known[CAMERA.code])

    def test_camera_caches_are_independent_even_with_same_local_identity(self):
        runtime, sync, detection, _ = self.make_runtime()
        other = replace(CAMERA, id="another-uuid", code="CAM_OTHER")
        sync.configs += (CameraConfiguration(other, ()),)
        runtime.pipeline.process.side_effect = lambda r, z, detection_indices: (
            VehicleRecognition(0, CAR, PlateRecognition("30A12345", .9)),)
        self.process(runtime, sync, detection, 1)
        detection.emit(1, camera=other)
        runtime._process_camera(other.code, sync.configs[1])
        self.assertEqual(runtime.pipeline.process.call_count, 2)
        self.assertEqual(runtime.drain(CAMERA.code)[0].camera, CAMERA)
        self.assertEqual(runtime.drain(other.code)[0].camera, other)

    def test_completed_plate_survives_a_busy_controller_without_another_ocr(self):
        runtime, sync, detection, _ = self.make_runtime()
        runtime.pipeline.process.return_value = (VehicleRecognition(0, CAR, PlateRecognition("30A12345", .9)),)
        self.process(runtime, sync, detection, 1)
        with patch("plate_ai.recognition_runtime.time.monotonic", return_value=time.monotonic()+10):
            result = runtime.drain(CAMERA.code)
        self.assertEqual(result[0].vehicles[0].license_plate.text, "30A12345")
        self.process(runtime, sync, detection, 2)
        self.assertEqual(runtime.pipeline.process.call_count, 1)

    def test_a_later_slow_crop_does_not_lose_an_already_completed_plate(self):
        runtime, sync, detection, state = self.make_runtime()
        state[:] = [((0, "track-1"), (1, "track-2")), frozenset({"track-1", "track-2"})]
        clock = [time.monotonic()]
        def process(result, zones, *, detection_indices):
            index = next(iter(detection_indices))
            if index == 1:
                clock[0] += 3.
            return (VehicleRecognition(index, CAR, PlateRecognition("30A12345", .9)),)
        runtime.pipeline.process.side_effect = process
        detection.emit(1, (CAR, CAR))
        clock[0] = detection.latest(CAMERA.code).frame.received_monotonic
        with patch("plate_ai.recognition_runtime.time.monotonic", side_effect=lambda:clock[0]):
            runtime._process_camera(CAMERA.code, sync.configs[0])
            results = runtime.drain(CAMERA.code)
        self.assertEqual([r.vehicles[0].object_key for r in results], ["track-1"])
        self.assertEqual(runtime._known[CAMERA.code], {"track-1"})

    def test_pipeline_filters_known_indices_before_any_plate_model_call(self):
        pipeline = object.__new__(RecognitionPipeline)
        pipeline.settings = RecognitionSettings()
        pipeline.plate, pipeline.characters = Mock(), Mock()
        pipeline.plate.detect.return_value = ()
        detection = Detection()
        result = detection.emit(1, (CAR, CAR))
        reads = pipeline.process(result, (), detection_indices=frozenset({1}))
        self.assertEqual([r.detection_index for r in reads], [1])
        self.assertEqual(pipeline.plate.detect.call_count, 1)
        pipeline.characters.detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
