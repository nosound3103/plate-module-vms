from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import threading
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock

from plate_ai.messaging.delivery import EventDelivery, EventDeliverySettings


def event(key, occurred=100., *, plate="30A-12345", kind=None, camera="camera-uuid", direction="IN"):
    payload = {"direction": direction, "confidence": .8,
               "vehicle": {"license_plate": {"text": plate, "status": "DETECTED"}}}
    if kind:
        payload.update(violation_type_code=kind, violation_evidence={"rule": "red_light_crossing"})
    return {"ai_modules": "PLATE", "camera_id": camera, "entity_type": "VEHICLE", "entity_id": key,
            "event_time": datetime.fromtimestamp(occurred, timezone(timedelta(hours=7))).isoformat(),
            "payload": payload, "snapshot_url": None, "snapshot_base64": None}


class OutboxTests(unittest.TestCase):
    def test_disabling_recognition_also_blocks_previously_pending_in_out_events(self):
        self.accept(event("pending-in-out"))
        self.accept(event("red-light", plate="99B12345", kind="RED_LIGHT"))
        self.delivery.recognition_enabled = False
        self.delivery._deliver(self.db, 1004.)
        self.assertEqual([e["entity_id"] for e in self.sent], ["red-light"])
        self.assertEqual(self.state("pending-in-out"), "PENDING")
        self.delivery.recognition_enabled = True
        self.delivery._deliver(self.db, 1006.)
        self.assertEqual([e["entity_id"] for e in self.sent], ["red-light", "pending-in-out"])

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.settings = EventDeliverySettings(state_path=str(Path(self.directory.name)/"events.sqlite3"))
        self.sent = []
        self.delivery = EventDelivery(self.settings, lambda e, **_: self.sent.append(e) or True)
        self.db = self.delivery._open()

    def tearDown(self):
        self.db.close()
        self.directory.cleanup()

    def accept(self, e, now=1000.):
        return self.delivery._accept(self.db, e["entity_id"], "CAM_TEST", json.dumps(e), now)

    def state(self, key):
        return self.db.execute("SELECT state FROM events WHERE entity_id=?", (key,)).fetchone()[0]

    def test_same_plate_changed_id_cooldown_and_exact_entity_retries(self):
        self.assertTrue(self.accept(event("one")))
        self.assertTrue(self.accept(event("one")))
        self.assertTrue(self.accept(event("id-changed", 101., plate="30a 12345")))
        self.assertEqual(self.state("id-changed"), "SUPPRESSED")
        self.delivery._deliver(self.db, 1004.)
        self.delivery._deliver(self.db, 1005.)
        self.assertEqual([e["entity_id"] for e in self.sent], ["one"])

    def test_configured_cooldown_expiry_different_camera_and_direction(self):
        self.delivery.settings = replace(self.settings, cooldown_seconds=2.)
        for e in (event("one"), event("after", 102.), event("other", camera="other-camera"),
                  event("out", direction="OUT")):
            self.assertTrue(self.accept(e))
            self.assertEqual(self.state(e["entity_id"]), "PENDING")

    def test_event_time_instead_of_publication_time_and_reordered_events(self):
        self.accept(event("one", 100.), 1000.)
        self.accept(event("delayed-duplicate", 99.), 1050.)
        self.accept(event("later-passage", 120.), 1050.)
        self.assertEqual(self.state("delayed-duplicate"), "SUPPRESSED")
        self.assertEqual(self.state("later-passage"), "PENDING")

    def test_unknown_plates_are_not_all_suppressed(self):
        self.accept(event("unknown-one", plate=""))
        self.accept(event("unknown-two", plate=""))
        self.assertEqual(self.state("unknown-two"), "PENDING")
        self.delivery._deliver(self.db, 1004.)
        self.delivery._deliver(self.db, 1004.1)  # ACK one; rate-limit UNKNOWN_PLATE too.
        self.delivery._deliver(self.db, 1005.5)
        self.assertEqual(len(self.sent), 2)

    def test_coalesces_normal_first_or_violation_first_without_changing_contract(self):
        for suffix, reverse in (("a", False), ("b", True)):
            normal = event("normal-"+suffix, 100. if not reverse else 120.)
            violation = event("red-"+suffix, 100.2 if not reverse else 120.2, kind="RED_LIGHT")
            before = json.dumps(violation, sort_keys=True)
            for e in ((violation, normal) if reverse else (normal, violation)):
                self.accept(e)
            self.assertEqual(self.state(normal["entity_id"]), "MERGED")
            self.assertEqual(self.state(violation["entity_id"]), "PENDING")
            stored = self.db.execute("SELECT body FROM events WHERE entity_id=?", (violation["entity_id"],)).fetchone()[0]
            self.assertEqual(json.dumps(json.loads(stored), sort_keys=True), before)

    def test_hold_allows_violation_to_arrive_before_recognition_published(self):
        self.accept(event("normal"))
        self.delivery._deliver(self.db, 1002.)
        self.assertFalse(self.sent)
        self.accept(event("red", 100.2, kind="RED_LIGHT"), 1002.)
        self.delivery._deliver(self.db, 1002.1)
        self.assertEqual([e["entity_id"] for e in self.sent], ["red"])

    def test_violation_is_not_suppressed_by_recognition_cooldown(self):
        self.accept(event("normal"))
        self.accept(event("red", 102., kind="RED_LIGHT"))
        self.assertEqual(self.state("normal"), "PENDING")
        self.assertEqual(self.state("red"), "PENDING")
        self.delivery._deliver(self.db, 1004.)
        self.delivery._deliver(self.db, 1004.1)
        self.delivery._deliver(self.db, 1005.5)
        self.assertEqual([e["entity_id"] for e in self.sent], ["red", "normal"])

    def test_late_violation_after_normal_handoff_is_logged_not_retimed(self):
        self.accept(event("normal"))
        self.delivery._deliver(self.db, 1004.)
        self.accept(event("red", 100.2, kind="RED_LIGHT"), 1004.1)
        self.assertEqual(self.state("normal"), "PENDING")  # Inflight cannot be retracted.
        self.delivery._deliver(self.db, 1004.2)
        with self.assertLogs("plate_ai.messaging.delivery", level="ERROR") as logs:
            self.delivery._deliver(self.db, 1005.6)
        self.assertIn("BACKEND_EVENT_DEDUP_REQUIRED", " ".join(logs.output))
        self.assertEqual(self.sent[-1]["event_time"], event("red", 100.2)["event_time"])

    def test_disconnect_failure_retry_and_no_duplicate_paho_message(self):
        publish, status = Mock(return_value=False), Mock(return_value=("PENDING", 0.))
        self.delivery.publish, self.delivery.status = publish, status
        self.accept(event("red", kind="RED_LIGHT"))
        self.delivery._deliver(self.db, 1000.)
        publish.return_value = True
        self.delivery._deliver(self.db, 1001.)
        for now in (1002., 1003., 1004.):
            self.delivery._deliver(self.db, now)
        self.assertEqual(publish.call_count, 2)
        status.return_value = ("ACK", 1004.)
        self.delivery._deliver(self.db, 1004.1)
        self.assertEqual(self.state("red"), "ACK")

    def test_restart_retains_cooldown_ack_and_replays_unacknowledged_same_entity(self):
        self.accept(event("one"))
        self.delivery._deliver(self.db, 1004.)
        self.delivery._deliver(self.db, 1004.1)
        self.accept(event("pending", 130., kind="RED_LIGHT"), 1005.)
        self.db.close()
        self.delivery = EventDelivery(self.settings, lambda e, **_: self.sent.append(e) or True)
        self.db = self.delivery._open()
        self.accept(event("different-track", 101.), 1006.)
        self.accept(event("one"), 1006.)
        self.assertEqual(self.state("different-track"), "SUPPRESSED")
        self.delivery._deliver(self.db, 1006.)
        self.assertEqual([e["entity_id"] for e in self.sent], ["one", "pending"])

    def test_expiry_limits_and_disabled_camera_do_not_crash_or_publish(self):
        self.delivery.settings = replace(self.settings, max_pending=1)
        self.accept(event("one"))
        self.assertFalse(self.accept(event("two", plate="OTHER")))
        self.delivery.eligible = lambda *_: False
        self.delivery._deliver(self.db, 1004.)
        self.assertFalse(self.sent)
        self.delivery._deliver(self.db, 1200.)
        self.assertEqual(self.state("one"), "EXPIRED")

    def test_wrong_timezone_and_base64_are_rejected(self):
        for e in (dict(event("wrong-tz"), event_time="2026-10-29T10:00:00+00:00"),
                  dict(event("base64"), snapshot_base64="image")):
            with self.assertRaises(ValueError):
                self.accept(e)

    def test_backend_legacy_dedup_keeps_violation_using_single_combined_payload(self):
        # Emulate the provided handler: camera/plate limiter + stored-event ±1s.
        backend_rows, cases, last_received = [], [], {}
        clock = [1000.]
        def backend(e, **_):
            identity = e["camera_id"], "".join(c for c in e["payload"]["vehicle"]["license_plate"]["text"].upper() if c.isalnum())
            occurred = datetime.fromisoformat(e["event_time"]).timestamp()
            if clock[0]-last_received.get(identity, -1e9) < 1:
                return True
            last_received[identity] = clock[0]
            if any(old_identity == identity and abs(old_occurred-occurred) <= 1 for old_identity, old_occurred in backend_rows):
                return True
            backend_rows.append((identity, occurred))
            if e["payload"].get("violation_type_code"):
                cases.append(e["entity_id"])
            return True
        self.delivery.publish = backend
        self.accept(event("normal"))
        self.delivery._deliver(self.db, 1001.)
        self.accept(event("red", 100.2, kind="RED_LIGHT"), 1001.)
        self.delivery._deliver(self.db, 1001.1)
        self.delivery._deliver(self.db, 1001.2)
        self.assertEqual(len(backend_rows), 1)
        self.assertEqual(cases, ["red"])


class WorkerTests(unittest.TestCase):
    def test_slow_disk_does_not_block_controller_submission(self):
        with TemporaryDirectory() as directory:
            settings = EventDeliverySettings(state_path=str(Path(directory)/"outbox.sqlite3"))
            with EventDelivery(settings, Mock(return_value=False)) as delivery:
                entered, release, returned = threading.Event(), threading.Event(), threading.Event()
                accept = delivery._accept
                def slow_accept(*args):
                    entered.set()
                    release.wait(2.)
                    return accept(*args)
                delivery._accept = slow_accept
                e = event("slow-disk")
                def submit():
                    delivery(e, camera_code="CAM_TEST", object_key="slow-disk")
                    returned.set()
                caller = threading.Thread(target=submit)
                try:
                    caller.start()
                    self.assertTrue(entered.wait(1.))
                    self.assertTrue(returned.wait(.5))
                    self.assertFalse(delivery(e, camera_code="CAM_TEST", object_key="slow-disk"))
                finally:
                    release.set()
                    caller.join(2.)

    def test_async_call_waits_for_disk_and_restores_outbox_after_restart(self):
        with TemporaryDirectory() as directory:
            settings = EventDeliverySettings(state_path=str(Path(directory)/"outbox.sqlite3"))
            e = event("durable")
            with EventDelivery(settings, Mock(return_value=False)) as delivery:
                deadline = time.monotonic()+3
                while not delivery(e, camera_code="CAM_TEST", object_key="durable"):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
            with sqlite3.connect(settings.state_path) as db:
                self.assertEqual(db.execute("SELECT state FROM events").fetchone()[0], "PENDING")
            with EventDelivery(settings, Mock(return_value=False)) as delivery:
                deadline = time.monotonic()+3
                while not delivery(e, camera_code="CAM_TEST", object_key="durable"):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
            with sqlite3.connect(settings.state_path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)

    def test_invalid_configuration_is_rejected(self):
        for values in ({"cooldown_seconds": 0}, {"cooldown_seconds": float("nan")},
                       {"state_path": ":memory:"}, {"max_pending": 0},
                       {"recognition_hold_seconds": 120}, {"max_pending": True},
                       {"retention_seconds": 1}):
            with self.assertRaises(ValueError):
                EventDeliverySettings(**values)


if __name__ == "__main__":
    unittest.main()
