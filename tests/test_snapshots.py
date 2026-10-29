from concurrent.futures import Future
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

from plate_ai.domain import Camera, CrossingLine
from plate_ai.detection import VehicleDetection
from plate_ai.passage_config import PassageSettings
from plate_ai.passage_runtime import PassageRuntime, _CameraState
from plate_ai.recognition import PlateRecognition, VehicleRecognition
from plate_ai.red_light import Violation, LightObservation
from plate_ai.red_light_config import RedLightSettings
from plate_ai.red_light_events import RedLightEventPublisher
from plate_ai.rtsp import CapturedFrame
from plate_ai.runtime import DetectionMetadata
from plate_ai.snapshot_config import SnapshotSettings
from plate_ai.snapshots import SnapshotService, StorageClient
from plate_ai.tracking import Track, VehicleTracker

CAMERA = Camera("e240a093-fae9-4c25-a11e-1aeb037a0a9b", "CAM_TEST", "rtsp://unused")
KEY = f"plate/2026-10-28/camera_{CAMERA.id}/actual-storage-id.jpg"


def response(data, status=200):
    result = Mock()
    result.status = status
    result.read.return_value = json.dumps(data).encode()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


def vehicle_track():
    vehicle = VehicleDetection("CAR", .9, (10, 10, 40, 60))
    track = Track(1, "actual-track", (.1, .1, .4, .6), vehicle, time.monotonic(),
                  direction="OUT", crossed_epoch=100., crossed_monotonic=time.monotonic()-3)
    track.remember(VehicleRecognition(0, vehicle, PlateRecognition("30A-12345", .85, (12, 45, 38, 58), .9)), captured_at=99., sequence=1, identity_epoch=0)
    return track


class StorageTests(unittest.TestCase):
    def test_multipart_contract_uses_object_key_not_presigned_url(self):
        opener = Mock(return_value=response({"success": True, "object_key": KEY,
                                            "url": "http://127.0.0.1/private?signature=secret"}))
        client = StorageClient("vms.example", SnapshotSettings(), opener=opener)
        self.assertEqual(client.upload(CAMERA, b"camera-jpeg-bytes", object_key="event-1"), KEY)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "https://vms.example/storage/upload/file")
        self.assertEqual(request.method, "POST")
        self.assertIn(b'name="camera_id"\r\n\r\n'+CAMERA.id.encode(), request.data)
        self.assertIn(b'name="category"\r\n\r\nplate', request.data)
        self.assertIn(b'name="file"; filename="snapshot.jpg"', request.data)
        self.assertIn(b"Content-Type: image/jpeg", request.data)
        self.assertIn(b"camera-jpeg-bytes", request.data)
        self.assertTrue(request.get_header("Content-type").startswith("multipart/form-data; boundary="))
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(opener.call_args.kwargs["timeout"], 4.)

    def test_timeout_and_500_retry_then_return_key(self):
        for failure in (TimeoutError(), URLError("offline"), HTTPError("unused", 500, "failure", {}, BytesIO())):
            opener = Mock(side_effect=[failure, response({"success": True, "object_key": KEY})])
            with patch("plate_ai.snapshots.time.sleep"):
                self.assertEqual(StorageClient("vms.example", SnapshotSettings(), opener=opener)
                                 .upload(CAMERA, b"jpeg", object_key="event"), KEY)
            self.assertEqual(opener.call_count, 2)

    def test_client_errors_do_not_retry_and_fall_back_without_image(self):
        for status in (403, 413, 422):
            opener = Mock(side_effect=HTTPError("unused", status, "failure", {}, BytesIO()))
            self.assertIsNone(StorageClient("vms.example", SnapshotSettings(), opener=opener)
                              .upload(CAMERA, b"jpeg", object_key="event"))
            self.assertEqual(opener.call_count, 1)

    def test_bad_responses_and_local_paths_are_never_used_as_snapshot_url(self):
        for data in ({"success": False, "object_key": KEY}, {"success": True}, [],
                     {"success": True, "object_key": "/snapshots/local.png"},
                     {"success": True, "object_key": "https://internal/a.jpg"},
                     {"success": True, "object_key": "plate/../secrets.jpg"},
                     {"success": True, "object_key": "plate/file.jpg?signature=secret"}):
            opener = Mock(return_value=response(data))
            self.assertIsNone(StorageClient("vms.example", SnapshotSettings(), opener=opener)
                              .upload(CAMERA, b"jpeg", object_key="event"))

    def test_image_limit_and_invalid_json(self):
        opener = Mock()
        client = StorageClient("vms.example", replace(SnapshotSettings(), max_bytes=3), opener=opener)
        self.assertIsNone(client.upload(CAMERA, b"four", object_key="event"))
        opener.assert_not_called()
        invalid = response({})
        invalid.read.return_value = b"invalid-json"
        opener.return_value = invalid
        self.assertIsNone(client.upload(CAMERA, b"jpg", object_key="event"))

    def test_settings_defaults_opt_out_validation(self):
        self.assertTrue(SnapshotSettings.from_env({}).enabled)
        self.assertFalse(SnapshotSettings.from_env({"PLATE_SNAPSHOT_ENABLED": "false"}).enabled)
        for env in ({"PLATE_SNAPSHOT_WORKERS": "0"}, {"PLATE_SNAPSHOT_TIMEOUT_SECONDS": "nan"},
                    {"PLATE_SNAPSHOT_JPEG_QUALITY": "101"}, {"PLATE_SNAPSHOT_ENABLED": "maybe"}):
            with self.assertRaises(ValueError):
                SnapshotSettings.from_env(env)


class AsyncSnapshotTests(unittest.TestCase):
    def test_jpeg_original_frame_local_evidence_and_no_mutation(self):
        with TemporaryDirectory() as directory:
            frame = np.zeros((80, 120, 3), np.uint8)
            frame[10:20, 20:40] = (0, 255, 0)
            original = frame.copy()
            client = Mock()
            client.upload.return_value = KEY
            with SnapshotService("unused", SnapshotSettings(), client=client) as snapshots:
                future = snapshots.submit(CAMERA, frame, object_key="event", local_dir=directory)
                self.assertEqual(future.result(timeout=3), KEY)
            camera, jpeg = client.upload.call_args.args
            self.assertEqual(camera, CAMERA)
            self.assertEqual(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape, frame.shape)
            np.testing.assert_array_equal(frame, original)
            files = list(Path(directory).glob("*.png"))
            self.assertEqual(len(files), 1)
            np.testing.assert_array_equal(cv2.imread(str(files[0])), original)

    def test_bounded_queue_never_waits_and_deadline_falls_back(self):
        entered, release = threading.Event(), threading.Event()
        def upload(*args, **kwargs):
            entered.set()
            release.wait(3)
            return KEY
        snapshots = SnapshotService("unused", replace(SnapshotSettings(), workers=1, max_pending=1),
                                    client=SimpleNamespace(upload=upload))
        try:
            first = snapshots.submit(CAMERA, np.zeros((10, 10, 3), np.uint8), object_key="first")
            self.assertTrue(entered.wait(2))
            second = snapshots.submit(CAMERA, np.zeros((10, 10, 3), np.uint8), object_key="second")
            self.assertTrue(second.done())
            self.assertIsNone(second.result())
            self.assertEqual(snapshots.resolve(first, 1.), (False, None))
            self.assertEqual(snapshots.resolve(first, 11.), (True, None))
            missing = snapshots.submit(CAMERA, None, object_key="missing")
            self.assertIsNone(missing.result())
        finally:
            release.set()
            first.result(timeout=3)
            snapshots.stop()


class SnapshotEventTests(unittest.TestCase):
    def test_in_out_images_mark_only_frozen_crossing_vehicle_with_direction_color(self):
        for direction, positions in (("IN", (.4, .44, .48, .54)), ("OUT", (.6, .56, .52, .46))):
            client = Mock()
            client.upload.return_value = KEY
            original = np.zeros((400, 400, 3), np.uint8)
            sync = SimpleNamespace(snapshot=lambda: SimpleNamespace(cameras=(SimpleNamespace(camera=CAMERA),)))
            detection = SimpleNamespace(frame_for=lambda m: CapturedFrame(m.sequence, m.captured_at,
                                                                          m.received_monotonic, original))
            with SnapshotService("unused", SnapshotSettings(), client=client) as storage:
                runtime = PassageRuntime(sync, detection, None, PassageSettings(), publish=Mock(), snapshots=storage)
                state = _CameraState(CAMERA, VehicleTracker(CAMERA.code, PassageSettings()),
                                     lines=(CrossingLine("line", ((.1, .5), (.9, .5))),))
                for index, y in enumerate(positions):
                    vehicle = VehicleDetection("CAR", .9, (160, round((y-.2)*400), 240, round(y*400)))
                    runtime._observe(state, DetectionMetadata(CAMERA, index, 100+index*.1,
                                                             time.monotonic(), 400, 400, (vehicle,)))
                key, (_, track) = next(iter(runtime._pending.items()))
                box = track.box
                self.assertEqual(track.direction, direction)
                track.box = (.75, .7, .95, .95)
                runtime._snapshot_futures[key].result(timeout=3)
                marked = cv2.imdecode(np.frombuffer(client.upload.call_args.args[1], np.uint8), cv2.IMREAD_COLOR)
                b, g, r = map(int, marked[round((box[1]+box[3])/2*399), round(box[0]*399)])
                if direction == "IN":
                    self.assertGreater(g, max(b, r)+60)
                    self.assertEqual((state.count_in, state.count_out), (1, 0))
                else:
                    self.assertGreater(r, 150)
                    self.assertGreater(g, 80)
                    self.assertLess(b, 60)
                    self.assertEqual((state.count_in, state.count_out), (0, 1))
                self.assertLess(int(marked[320, 300].max()), 20)
                self.assertFalse(original.any())

    def test_disabled_passage_events_do_not_queue_snapshots_or_publish_but_still_count(self):
        storage, publish = Mock(), Mock()
        runtime = PassageRuntime(SimpleNamespace(), None, None, PassageSettings(),
                                  publish=publish, snapshots=storage, emit_events=False)
        state = _CameraState(CAMERA, VehicleTracker(CAMERA.code, PassageSettings()),
                             lines=(CrossingLine("line", ((.1, .5), (.9, .5))),))
        for index, y in enumerate((.4, .44, .48, .54)):
            vehicle = VehicleDetection("CAR", .9, (40, round((y-.2)*100), 60, round(y*100)))
            runtime._observe(state, DetectionMetadata(CAMERA, index, 100+index*.1,
                                                     time.monotonic(), 100, 100, (vehicle,)))
        self.assertEqual((state.count_in, state.count_out), (1, 0))
        self.assertFalse(runtime._pending)
        storage.submit.assert_not_called()
        publish.assert_not_called()

    def test_normal_crossing_upload_then_mqtt_retry_reuses_image_and_event(self):
        client = Mock()
        client.upload.return_value = KEY
        sent, online = [], [False]
        def publish(event, **kwargs):
            if not online[0]:
                return False
            sent.append(event)
            return True
        with SnapshotService("unused", SnapshotSettings(), client=client) as snapshots:
            sync = SimpleNamespace(snapshot=lambda: SimpleNamespace(cameras=(SimpleNamespace(camera=CAMERA),)))
            frame = np.zeros((100, 100, 3), np.uint8)
            detection = SimpleNamespace(frame_for=lambda m: CapturedFrame(m.sequence, m.captured_at,
                                                                         m.received_monotonic, frame))
            runtime = PassageRuntime(sync, detection, None, PassageSettings(), publish=publish, snapshots=snapshots)
            state = _CameraState(CAMERA, VehicleTracker(CAMERA.code, PassageSettings()),
                                 lines=(CrossingLine("line", ((.1, .5), (.9, .5))),))
            now = time.monotonic()
            for i, y in enumerate((.4, .44, .48, .54)):
                vehicle = VehicleDetection("CAR", .9, (40, int((y-.2)*100), 60, int(y*100)))
                runtime._observe(state, DetectionMetadata(CAMERA, i, 100+i*.1, now+i*.1, 100, 100, (vehicle,)))
            self.assertEqual(state.count_in, 1)
            key, (_, track) = next(iter(runtime._pending.items()))
            track.remember(VehicleRecognition(0, track.vehicle, PlateRecognition("30A-12345", .85)))
            runtime._snapshot_futures[key].result(timeout=3)
            runtime._flush(now+3)
            self.assertEqual(track.event["snapshot_url"], KEY)
            online[0] = True
            runtime._flush(now+5)
            runtime._flush(now+6)
            self.assertEqual(client.upload.call_count, 1)
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0]["snapshot_url"], KEY)
            self.assertIsNone(sent[0]["snapshot_base64"])
            self.assertEqual(sent[0]["payload"]["direction"], "IN")

    def test_normal_event_does_not_wait_forever_or_change_event_time(self):
        sync = SimpleNamespace(snapshot=lambda: SimpleNamespace(cameras=(SimpleNamespace(camera=CAMERA),)))
        with SnapshotService("unused", SnapshotSettings(), client=Mock()) as snapshots:
            sent = []
            runtime = PassageRuntime(sync, None, None, PassageSettings(), snapshots=snapshots,
                                     publish=lambda e, **_: sent.append(e) or True)
            track = vehicle_track()
            runtime._pending[track.object_key] = (CAMERA, track)
            runtime._snapshot_futures[track.object_key] = Future()
            runtime._flush(track.crossed_monotonic+11)
            self.assertEqual(len(sent), 1)
            self.assertIsNone(sent[0]["snapshot_url"])
            self.assertTrue(sent[0]["event_time"].startswith("1970-01-01"))

    def test_violation_object_key_and_failure_fallback_preserve_crossing(self):
        for original_result, annotated_result in ((KEY, "plate/marked.jpg"), (KEY, None), (None, KEY), (None, None)):
            upload_result = (annotated_result or original_result) if original_result else None
            with TemporaryDirectory() as directory:
                track, sent = vehicle_track(), []
                client = Mock()
                client.upload.side_effect = [original_result, annotated_result]
                with SnapshotService("unused", SnapshotSettings(), client=client) as snapshots:
                    publisher = RedLightEventPublisher(replace(RedLightSettings(), snapshot_dir=directory),
                                                      lambda event, **_: sent.append(event) or True, snapshots=snapshots)
                    try:
                        publisher.submit(CAMERA, Violation(track, 100., .8, np.zeros((10, 10, 3), np.uint8), 100.1, 1, LightObservation("RED", .8, {}, 100.1), ((0, 0), (1, 0), (1, 1), (0, 1))))
                        item = next(iter(publisher.pending.values()))
                        files = item[3].result(timeout=3)
                        publisher.flush(item[2]+3)
                        item[7].result(timeout=3).result(timeout=3)
                        publisher.flush(item[2]+3)
                        item[6].result(timeout=3)
                        publisher.flush(item[2]+4)
                        self.assertEqual(len(sent), 1)
                        self.assertEqual(sent[0]["snapshot_url"], upload_result)
                        self.assertEqual(sent[0]["payload"]["violation_type_code"], "RED_LIGHT")
                        self.assertEqual(sent[0]["payload"]["direction"], "OUT")
                        self.assertEqual(sent[0]["event_time"], sent[0]["payload"]["violation_evidence"]["crossing_timestamp"])
                        self.assertEqual(client.upload.call_count, 2)
                        if upload_result:
                            evidence = sent[0]["payload"]["violation_evidence"]
                            self.assertEqual(evidence["original_snapshot_url"], original_result)
                            self.assertEqual(evidence["annotated_snapshot_url"], annotated_result)
                            self.assertEqual(evidence["snapshot_view"], "ANNOTATED" if annotated_result else "ORIGINAL")
                    finally:
                        publisher.stop()


if __name__ == "__main__":
    unittest.main()
