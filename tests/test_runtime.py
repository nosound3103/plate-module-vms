import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from plate_ai.detection.config import DetectionSettings
from plate_ai.core.domain import Camera
from plate_ai.camera.stream import CapturedFrame, LatestFrameStream
from plate_ai.detection.runtime import DetectionRuntime, DetectionResult


def camera(source="rtsp://unused/first", code="CAM_01"):
    return Camera("e240a093-fae9-4c25-a11e-1aeb037a0a9b", code, source)


def wait_for(predicate, timeout=3.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Runtime condition was not reached")


class FakeStream:
    created = []

    def __init__(self, camera, settings):
        self.camera, self.alive, self.sequence = camera, False, 0
        self.created.append(self)

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def request_stop(self):
        self.alive = False

    def stop(self, timeout=0.):
        self.request_stop()
        return True

    def latest(self):
        self.sequence += 1
        return CapturedFrame(self.sequence, time.time(), time.monotonic(), self.camera.source)


class FakeSync:
    def __init__(self, cameras):
        self.cameras = tuple(cameras)

    def snapshot(self):
        return SimpleNamespace(cameras=tuple(SimpleNamespace(camera=c) for c in self.cameras))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        FakeStream.created = []

    def test_source_change_discards_inflight_result_and_replaces_reader(self):
        entered, release = threading.Event(), threading.Event()

        class Detector:
            def detect(self, frame):
                if frame.endswith("first"):
                    entered.set()
                    release.wait(3.)
                return ()

        sync = FakeSync([camera()])
        runtime = DetectionRuntime(sync, DetectionSettings(), detector_factory=lambda _: Detector(), stream_factory=FakeStream)
        runtime.start()
        try:
            self.assertTrue(entered.wait(2.))
            sync.cameras = (camera("rtsp://unused/second"),)
            wait_for(lambda: len(FakeStream.created) == 2)
            self.assertFalse(FakeStream.created[0].alive)
            self.assertIsNone(runtime.latest("CAM_01"))
            release.set()
            wait_for(lambda: runtime.latest("CAM_01") is not None)
            self.assertEqual(runtime.latest("CAM_01").camera.source, "rtsp://unused/second")
            sync.cameras = ()
            wait_for(lambda: not FakeStream.created[1].alive)
            self.assertIsNone(runtime.latest("CAM_01"))
        finally:
            release.set()
            self.assertTrue(runtime.stop())

    def test_shared_detector_services_two_cameras_and_loads_once(self):
        calls, loads = [], []

        class Detector:
            def detect(self, frame):
                calls.append(frame)
                return ()

        def factory(settings):
            loads.append(settings)
            return Detector()

        sync = FakeSync([camera(), camera("rtsp://unused/other", "CAM_02")])
        with DetectionRuntime(sync, DetectionSettings(), detector_factory=factory, stream_factory=FakeStream) as runtime:
            wait_for(lambda: runtime.latest("CAM_01") is not None and runtime.latest("CAM_02") is not None)
            self.assertEqual(len(loads), 1)
            self.assertIn("rtsp://unused/first", calls)
            self.assertIn("rtsp://unused/other", calls)
        self.assertTrue(all(not reader.alive for reader in FakeStream.created))

    def test_old_results_are_not_returned(self):
        runtime = DetectionRuntime(FakeSync([]), DetectionSettings(), stream_factory=FakeStream)
        frame = CapturedFrame(1, time.time()-2, time.monotonic()-2, None)
        runtime._results["CAM_01"] = DetectionResult(camera(), frame, (), 1.)
        self.assertIsNone(runtime.latest("CAM_01"))


class RtspTests(unittest.TestCase):
    def test_latest_queue_is_bounded_and_reader_releases_capture(self):
        captures = []

        class Capture:
            def __init__(self, source, backend, params):
                self.params, self.released = params, False
                captures.append(self)

            def isOpened(self):
                return True

            def read(self):
                time.sleep(.005)
                return True, np.zeros((10, 10, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        with patch("cv2.VideoCapture", Capture):
            stream = LatestFrameStream(camera(), DetectionSettings())
            stream.start()
            try:
                wait_for(lambda: stream.frames.qsize() == 1)
                time.sleep(.04)
                self.assertEqual(stream.frames.qsize(), 1)
                frame = stream.latest()
                self.assertGreater(frame.sequence, 1)
                self.assertLess(time.monotonic()-frame.received_monotonic, .2)
            finally:
                self.assertTrue(stream.stop(1.))
        self.assertTrue(captures[0].released)


if __name__ == "__main__":
    unittest.main()
