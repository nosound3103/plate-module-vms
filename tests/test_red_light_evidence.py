from dataclasses import replace
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
import cv2
import numpy as np

from plate_ai.core.domain import Camera
from plate_ai.detection.detector import VehicleDetection
from plate_ai.tracking.config import PassageSettings
from plate_ai.recognition.pipeline import PlateRecognition, VehicleRecognition
from plate_ai.traffic_light.detector import LightObservation, RedLightViolationDetector, Violation
from plate_ai.traffic_light.config import RedLightSettings
from plate_ai.traffic_light.evidence import prepare_evidence, upload_original, upload_evidence, draw_vehicle_highlight
from plate_ai.traffic_light.events import build_red_light_event, RedLightEventPublisher
from plate_ai.detection.runtime import DetectionMetadata
from plate_ai.tracking.tracker import Track, VehicleTracker

CAMERA = Camera("e240a093-fae9-4c25-a11e-1aeb037a0a9b", "CAM_TEST", "rtsp://unused")
STOP = ((.1, .1), (.9, .1), (.9, .5), (.1, .5))
ZONE = ((.1, .5), (.9, .5), (.9, .9), (.1, .9))
LIGHT = ((0., 0.), (.2, 0.), (.2, .2), (0., .2))


def track():
    return Track(1, "source-track", (.4, .3, .6, .5), VehicleDetection("CAR", .9, (40, 30, 60, 50)), 100.)


def remember(t, *, confidence=.9, box=(42, 40, 58, 48), timestamp=99., text="30A12345"):
    read = VehicleRecognition(0, t.vehicle, PlateRecognition(text, confidence, box, .95))
    t.remember(read, captured_at=timestamp, sequence=11, identity_epoch=t.identity_epoch)


def violation(t):
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[3:15, 3:15] = (0, 0, 255)
    return Violation(t, 100., .9, frame, 100.1, 12,
                     LightObservation("RED", .9, {"RED": .2, "GREEN": 0.}, 100.1), LIGHT, STOP, ZONE)


class EvidenceTests(unittest.TestCase):
    def test_rejected_crossing_reports_signal_reason(self):
        detector = RedLightViolationDetector(STOP, ZONE, RedLightSettings())
        t = track()
        with self.assertLogs("plate_ai.traffic_light.detector", level="INFO") as captured:
            for i, y in enumerate((.4, .44, .48, .54, .57)):
                t.box = (.4, y-.2, .6, y)
                now = 100.+i*.1
                self.assertFalse(detector.update([t], LightObservation("GREEN", .9, {}, now), now,
                                                np.zeros((100, 100, 3), np.uint8), frame_sequence=i+1))
        self.assertEqual(sum("CROSSING_REJECTED" in line for line in captured.output), 1)
        self.assertTrue(any("reason=SIGNAL_NOT_RED" in line for line in captured.output))

    def test_highlight_uses_crossing_box_and_uploads_original_and_marked_copy(self):
        t = track()
        remember(t)
        v = violation(t)
        # Live track has moved; the highlighted box must stay at the crossing position.
        t.box = (.7, .7, .9, .9)
        with TemporaryDirectory() as directory:
            files = prepare_evidence(CAMERA, v, directory)
            uploads = []
            def submit(camera, image, *, object_key):
                uploads.append((image.copy(), object_key))
                future = Future()
                future.set_result("plate/original.jpg" if object_key.endswith(":original") else "plate/marked.jpg")
                return future
            service = Mock(submit=submit)
            result = upload_evidence(CAMERA, files, service, "event", v, "30A12345").result(timeout=1)
            self.assertEqual(result, {"original": "plate/original.jpg", "annotated": "plate/marked.jpg"})
            np.testing.assert_array_equal(uploads[0][0], v.frame)
            self.assertFalse(np.array_equal(uploads[1][0], v.frame))
            marked = np.zeros((1000, 1000, 3), np.uint8)
            draw_vehicle_highlight(marked, v, "30A12345")
            np.testing.assert_array_equal(marked[400, 400], (0, 0, 255))
            np.testing.assert_array_equal(marked[800, 700], (0, 0, 0))

    def test_freezes_vehicle_bbox_plate_and_id_even_when_track_changes_after_crossing(self):
        t = track()
        remember(t)
        v = violation(t)
        t.box = (.7, .6, .9, .9)
        t.vehicle = VehicleDetection("MOTORBIKE", .8, (70, 60, 90, 90))
        t.id, t.object_key, t.identity_epoch = 2, "later-track", 1
        t.votes.clear()
        remember(t, box=(72, 80, 88, 88), text="99A99999")
        frozen = replace(v, frame=None)
        e = build_red_light_event(CAMERA, frozen, None)
        self.assertEqual(e["entity_id"], "source-track:RED_LIGHT")
        self.assertEqual(e["payload"]["vehicle"]["vehicle_type"], "CAR")
        self.assertEqual(e["payload"]["vehicle"]["license_plate"]["text"], "30A12345")
        self.assertEqual(e["payload"]["violation_evidence"]["vehicle_bbox"], [.4, .3, .6, .5])
        self.assertEqual(e["payload"]["violation_evidence"]["track_id"], 1)

    def test_pending_ocr_requires_same_identity_and_bounded_source_time(self):
        for changed_epoch, timestamp, expected in ((False, 101., True), (True, 101., False), (False, 105., False)):
            t = track()
            v = violation(t)
            t.identity_epoch += int(changed_epoch)
            remember(t, timestamp=timestamp)
            self.assertEqual(build_red_light_event(CAMERA, v, None) is not None, expected)

    def test_rejects_low_confidence_outside_crop_and_ocr_without_frame_proof(self):
        for mode in ("low-confidence", "outside-vehicle", "unproven"):
            t = track()
            if mode == "unproven":
                t.remember(VehicleRecognition(0, t.vehicle, PlateRecognition("30A12345", .99, (42, 40, 58, 48), .99)))
            else:
                remember(t, confidence=.5 if mode == "low-confidence" else .99,
                         box=(0, 0, 20, 20) if mode == "outside-vehicle" else (42, 40, 58, 48))
            self.assertIsNone(build_red_light_event(CAMERA, violation(t), None), mode)

    def test_rejects_green_unknown_stale_or_missing_signal_even_with_valid_plate(self):
        t = track()
        remember(t)
        v = violation(t)
        for signal in (None, LightObservation("GREEN", .9, {}, 100.1),
                       LightObservation("UNKNOWN", 0., {}, 100.1), LightObservation("RED", .9, {}, 99.)):
            self.assertIsNone(build_red_light_event(CAMERA, replace(v, signal=signal), None))

    def test_original_and_annotated_are_separate_and_upload_receives_original(self):
        t = track()
        remember(t)
        v = violation(t)
        before = v.frame.copy()
        with TemporaryDirectory() as directory:
            upload = Mock()
            files = prepare_evidence(CAMERA, v, directory)
            upload_original(CAMERA, files, upload, "source-track:RED_LIGHT")
            np.testing.assert_array_equal(cv2.imread(files.original), before)
            np.testing.assert_array_equal(v.frame, before)
            self.assertFalse(np.array_equal(cv2.imread(files.annotated), before))
            np.testing.assert_array_equal(upload.submit.call_args.args[1], before)
            self.assertEqual(files.frame_sha256, hashlib.sha256(before.tobytes()).hexdigest())
            data = json.loads(Path(files.manifest).read_text(encoding="utf-8"))
            self.assertEqual(data["entity_id"], "source-track:RED_LIGHT")
            self.assertEqual(data["frame_sequence"], 12)
            self.assertEqual(data["vehicle_bbox"], [.4, .3, .6, .5])

    def test_stationary_bbox_jitter_beyond_old_deadband_does_not_confirm_crossing(self):
        detector = RedLightViolationDetector(STOP, ZONE, RedLightSettings(), light_points=LIGHT)
        t, events = track(), []
        for i, y in enumerate((.48, .49, .501, .509, .511, .5, .508)):
            t.box = (.4, y-.2, .6, y)
            now = 100.+i*.1
            events.extend(detector.update([t], LightObservation("RED", .9, {}, now), now,
                                          np.zeros((100, 100, 3), np.uint8), frame_sequence=i+1))
        self.assertFalse(events)

    def test_confirmation_on_green_keeps_red_crossing_pixels_and_geometry(self):
        detector = RedLightViolationDetector(STOP, ZONE, RedLightSettings(), light_points=LIGHT)
        t, events = track(), []
        image = np.zeros((100, 100, 3), np.uint8)
        for i, y in enumerate((.4, .44, .48, .52, .56)):
            t.box = (.4, y-.2, .6, y)
            image[:] = (0, 0, 255) if i < 4 else (0, 255, 0)
            now = 100.+i*.1
            events.extend(detector.update([t], LightObservation("RED" if i < 4 else "GREEN", .9, {}, now),
                                          now, image, frame_sequence=i+1))
        self.assertEqual(len(events), 1)
        v = events[0]
        np.testing.assert_array_equal(v.frame[0, 0], (0, 0, 255))
        self.assertEqual(v.frame_sequence, 4)
        self.assertEqual(v.vehicle_box[-1], .52)
        self.assertEqual(v.signal.state, "RED")

    def test_identity_break_during_crossing_cancels_candidate(self):
        detector = RedLightViolationDetector(STOP, ZONE, RedLightSettings())
        t, events = track(), []
        for i, y in enumerate((.4, .44, .48, .52, .56)):
            t.box = (.4, y-.2, .6, y)
            if i == 4:
                t.identity_epoch += 1
            now = 100.+i*.1
            events.extend(detector.update([t], LightObservation("RED", .9, {}, now), now,
                                          np.zeros((100, 100, 3), np.uint8)))
        self.assertFalse(events)

    def test_red_boundary_jitter_cannot_be_confirmed_by_later_movement_on_green(self):
        detector = RedLightViolationDetector(STOP, ZONE, RedLightSettings())
        t, events = track(), []
        for i, y in enumerate((.48, .49, .501, .54, .57)):
            t.box = (.4, y-.2, .6, y)
            now = 100.+i*.1
            events.extend(detector.update([t], LightObservation("RED" if i < 3 else "GREEN", .9, {}, now),
                                          now, np.zeros((100, 100, 3), np.uint8)))
        self.assertFalse(events)

    def test_unverified_plate_never_uploads_snapshot(self):
        t = track()
        remember(t, confidence=.5)
        snapshots = Mock()
        with TemporaryDirectory() as directory:
            publisher = RedLightEventPublisher(replace(RedLightSettings(), snapshot_dir=directory), Mock(),
                                              snapshots=snapshots)
            try:
                publisher.submit(CAMERA, violation(t))
                item = next(iter(publisher.pending.values()))
                item[3].result(timeout=3)
                publisher.flush(item[2]+3)
                snapshots.submit.assert_not_called()
                publisher.publish.assert_not_called()
            finally:
                publisher.stop()

    def test_ambiguous_tracker_match_invalidates_identity_without_altering_counting_fields(self):
        tracker = VehicleTracker(CAMERA.code, PassageSettings())
        car = VehicleDetection("CAR", .9, (40, 30, 60, 50))
        first = DetectionMetadata(CAMERA, 1, 100., 100., 100, 100, (car,))
        t = tracker.update(first)[0]
        t.direction, t.crossed_epoch = "IN", 100.
        tracker.update(replace(first, sequence=2, received_monotonic=100.1, detections=(car, car)))
        self.assertGreater(t.identity_epoch, 0)
        self.assertFalse(t.association_reliable)
        self.assertEqual((t.direction, t.crossed_epoch), ("IN", 100.))


if __name__ == "__main__":
    unittest.main()
