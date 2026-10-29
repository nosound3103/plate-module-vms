import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from plate_ai.detection import VehicleDetector
from plate_ai.detection_config import DetectionSettings


class ConfigurationTests(unittest.TestCase):
    def test_defaults_do_not_require_ai_environment(self):
        settings = DetectionSettings.from_env({})
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.max_fps, 5.)
        self.assertEqual(settings.confidence, .4)

    def test_explicit_device_and_thresholds(self):
        settings = DetectionSettings.from_env({"PLATE_DEVICE": "cuda:0", "PLATE_MAX_FPS": "8",
                                               "PLATE_VEHICLE_CONFIDENCE": "0.6"})
        self.assertEqual((settings.device, settings.max_fps, settings.confidence), ("cuda:0", 8., .6))

    def test_engine_requires_cuda_and_verified_filename(self):
        for values in ({"vehicle_engine":"vehicle.engine"},
                       {"vehicle_engine":"../vehicle.engine", "vehicle_engine_sha256":"a"*64, "device":"cuda:0"},
                       {"vehicle_engine":"vehicle.engine", "vehicle_engine_sha256":"a"*64}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DetectionSettings(**values)

    def test_invalid_limits_are_rejected(self):
        for name, value in (("PLATE_MAX_FPS", "nan"), ("PLATE_IMAGE_SIZE", "31"),
                            ("PLATE_VEHICLE_CONFIDENCE", "0"), ("PLATE_RTSP_READ_TIMEOUT_MS", "-1"),
                            ("PLATE_DEVICE", "cuda:no"), ("PLATE_VEHICLE_SHA256", "bad")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                DetectionSettings.from_env({name: value})


class InferenceTests(unittest.TestCase):
    def synthetic_detector(self, predictions):
        detector = VehicleDetector.__new__(VehicleDetector)
        detector.settings = DetectionSettings()
        detector.device, detector.torch = torch.device("cpu"), torch
        detector.classes = {0: "CAR", 1: "MOTORBIKE"}
        detector.stride, detector.image_size = 32, 640
        detector.model = lambda image: predictions.to(image.device)
        return detector

    def test_nonempty_prediction_nms_confidence_and_classes(self):
        prediction = torch.tensor([[[320, 320, 200, 200, .9, .9, .1],
                                    [322, 322, 200, 200, .8, .9, .1],
                                    [100, 100, 50, 50, .9, .1, .95]]])
        detections = self.synthetic_detector(prediction).detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 2)
        self.assertEqual({d.vehicle_type for d in detections}, {"CAR", "MOTORBIKE"})
        car = next(d for d in detections if d.vehicle_type == "CAR")
        self.assertEqual(car.bbox, (220, 220, 420, 420))
        self.assertAlmostEqual(car.confidence, .81, places=5)

    def test_letterbox_rescales_to_source_and_clamps(self):
        prediction = torch.tensor([[[320, 192, 200, 100, .9, .9, .1],
                                    [20, 20, 200, 200, .9, .1, .9]]])
        detections = self.synthetic_detector(prediction).detect(np.zeros((360, 640, 3), dtype=np.uint8))
        car = next(d for d in detections if d.vehicle_type == "CAR")
        self.assertEqual(car.bbox, (220, 130, 420, 230))
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            self.assertTrue(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360)

    def test_no_detection_is_empty_tuple(self):
        detector = self.synthetic_detector(torch.zeros((1, 1, 7)))
        self.assertEqual(detector.detect(np.zeros((100, 200, 3), dtype=np.uint8)), ())

    def test_static_engine_letterbox_maps_back_to_landscape_frame(self):
        detector = self.synthetic_detector(torch.tensor([[[320, 320, 200, 100, .9, .9, .1]]]))
        detector.square_input = True
        def forward(image):
            self.assertEqual(tuple(image.shape), (1, 3, 640, 640))
            return torch.tensor([[[320, 320, 200, 100, .9, .9, .1]]])
        detector.model = forward
        self.assertEqual(detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))[0].bbox,
                         (220, 130, 420, 230))

    def test_static_engine_letterbox_maps_back_to_portrait_frame(self):
        detector = self.synthetic_detector(torch.tensor([[[320, 320, 100, 200, .9, .9, .1]]]))
        detector.square_input = True
        self.assertEqual(detector.detect(np.zeros((640, 360, 3), dtype=np.uint8))[0].bbox,
                         (130, 220, 230, 420))

    def test_checksum_mismatch_rejected_before_unpickle(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, DetectionSettings.vehicle_weight).write_bytes(b"invalid checkpoint")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                VehicleDetector(DetectionSettings(weights_dir=Path(directory)))


class ModelSmokeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("PLATE_VEHICLE_ENGINE") and torch.cuda.is_available(), "TensorRT engine is not configured")
    def test_configured_engine_cuda_forward(self):
        from plate_ai.tensorrt_runtime import TensorRTEngine
        detector = VehicleDetector(DetectionSettings.from_env())
        self.assertIsInstance(detector.model, TensorRTEngine)
        self.assertTrue(detector.square_input)
        for shape in ((360, 640, 3), (640, 360, 3)):
            self.assertIsInstance(detector.detect(np.zeros(shape, dtype=np.uint8)), tuple)
        torch.cuda.synchronize()

    def test_checkpoint_load_and_cpu_forward(self):
        directory = Path(os.environ.get("PLATE_TEST_WEIGHTS", "/models"))
        self.assertTrue((directory/DetectionSettings.vehicle_weight).is_file(), "Provide the model artifact for smoke tests")
        detector = VehicleDetector(DetectionSettings(weights_dir=directory))
        self.assertEqual(detector.classes, {0: "CAR", 1: "MOTORBIKE"})
        detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertIsInstance(detections, tuple)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_checkpoint_load_and_cuda_forward(self):
        directory = Path(os.environ.get("PLATE_TEST_WEIGHTS", "/models"))
        detector = VehicleDetector(DetectionSettings(device="cuda:0", weights_dir=directory))
        detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
        torch.cuda.synchronize()
        self.assertEqual(detector.device.type, "cuda")
        self.assertIsInstance(detections, tuple)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
