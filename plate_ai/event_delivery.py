"""Durable, bounded PLATE outbox and per-event-kind passage cooldown.

Disk I/O and delivery run on a worker, never on the detection/bbox controller.
PUBACK means broker receipt only; backend entity-aware idempotency is still
required for the crash window between broker acceptance and recording its ACK.
"""
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import queue
import sqlite3
import threading
import time

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventDeliverySettings:
    state_path: str = "snapshots/event_delivery.sqlite3"
    cooldown_seconds: float = 10.
    recognition_hold_seconds: float = 3.
    ttl_seconds: float = 120.
    retention_seconds: float = 86400.
    max_pending: int = 128
    max_records: int = 10000

    def __post_init__(self):
        if not isinstance(self.state_path, str) or not self.state_path.strip() or self.state_path == ":memory:":
            raise ValueError("PLATE_EVENT_STATE_PATH must name a persistent file")
        for name in ("cooldown_seconds", "recognition_hold_seconds", "ttl_seconds", "retention_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"PLATE_EVENT_{name.upper()} must be positive and finite")
        for name in ("max_pending", "max_records"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"PLATE_EVENT_{name.upper()} must be a positive integer")
        if self.retention_seconds < max(self.cooldown_seconds, self.ttl_seconds) or self.max_records < self.max_pending:
            raise ValueError("PLATE event retention/capacity must cover cooldown and pending delivery")
        if self.recognition_hold_seconds >= self.ttl_seconds:
            raise ValueError("PLATE event recognition hold must be shorter than delivery TTL")

    @classmethod
    def from_env(cls, environ=None):
        import os
        env = os.environ if environ is None else environ
        try:
            return cls(
                state_path=env.get("PLATE_EVENT_STATE_PATH", cls.state_path),
                cooldown_seconds=float(env.get("PLATE_EVENT_COOLDOWN_SECONDS", cls.cooldown_seconds)),
                recognition_hold_seconds=float(env.get("PLATE_EVENT_RECOGNITION_HOLD_SECONDS", cls.recognition_hold_seconds)),
                ttl_seconds=float(env.get("PLATE_EVENT_DELIVERY_TTL_SECONDS", cls.ttl_seconds)),
                retention_seconds=float(env.get("PLATE_EVENT_RETENTION_SECONDS", cls.retention_seconds)),
                max_pending=int(env.get("PLATE_EVENT_OUTBOX_MAX_PENDING", cls.max_pending)),
                max_records=int(env.get("PLATE_EVENT_STATE_MAX_RECORDS", cls.max_records)),
            )
        except (ValueError, TypeError):
            raise ValueError("Invalid PLATE event delivery configuration") from None


class EventDelivery:
    manages_delivery = True

    def __init__(self, settings, publish, status=None, *, eligible=None, guard_seconds=1.25,
                 recognition_enabled=True):
        self.settings, self.publish, self.status = settings, publish, status
        self.eligible, self.guard = eligible, guard_seconds
        self.recognition_enabled = recognition_enabled
        self._queue = queue.Queue(settings.max_pending)
        self._lock = threading.Lock()
        self._results = OrderedDict()
        self._submitted = set()
        self._stop, self._wake, self._ready = threading.Event(), threading.Event(), threading.Event()
        self._error = None
        self._inflight, self._retry_at = {}, {}
        self._thread = threading.Thread(target=self._run, name="plate-event-delivery", daemon=True)

    def __enter__(self):
        self._thread.start()
        if not self._ready.wait(5.) or self._error:
            self.stop()
            raise RuntimeError("PLATE event state could not be opened") from self._error
        return self

    def __exit__(self, *_):
        self.stop()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread.ident is not None:
            self._thread.join(5.)
        if self._thread.is_alive():
            LOGGER.error("PLATE outbox result=SHUTDOWN_TIMEOUT")

    def __call__(self, event, *, camera_code, object_key):
        """Report success only after durable acceptance or logged suppression.

        The controller retries until the worker acknowledges persistence. An
        existing entity ID is immutable, including across MQTT retries/restart.
        """
        with self._lock:
            if self._stop.is_set() or self._error:
                return False
            if object_key in self._results:
                return self._results[object_key]
            if object_key in self._submitted or len(self._submitted) >= self.settings.max_pending:
                return False
            # Freeze before handing ownership to the worker; no camera frame.
            body = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            self._submitted.add(object_key)
            self._queue.put_nowait((object_key, camera_code, body))
        self._wake.set()
        return False

    def _result(self, key, accepted):
        with self._lock:
            self._submitted.discard(key)
            if accepted:
                self._results[key] = True
                while len(self._results) > 1024:
                    self._results.popitem(last=False)

    def _open(self):
        path = Path(self.settings.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=.1)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("""CREATE TABLE IF NOT EXISTS events (
            entity_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, camera_code TEXT NOT NULL,
            plate TEXT NOT NULL, kind TEXT NOT NULL, direction TEXT NOT NULL,
            occurred REAL NOT NULL, created REAL NOT NULL, state TEXT NOT NULL,
            ack_at REAL, body TEXT NOT NULL)""")
        db.execute("CREATE INDEX IF NOT EXISTS passage ON events(camera_id, plate, kind, direction, occurred)")
        db.commit()
        return db

    def _accept(self, db, key, code, body, now):
        from .events import normalize_plate
        event = json.loads(body)
        payload = event["payload"]
        moment = datetime.fromisoformat(event["event_time"])
        if (event.get("ai_modules") != "PLATE" or event.get("entity_type") != "VEHICLE"
                or event.get("entity_id") != key or not event.get("camera_id")
                or moment.utcoffset() is None or moment.utcoffset().total_seconds() != 7*3600
                or event.get("snapshot_base64") is not None):
            raise ValueError("Invalid PLATE event contract")
        if db.execute("SELECT 1 FROM events WHERE entity_id=?", (key,)).fetchone():
            LOGGER.info("PLATE camera=%s object_key=%s result=ENTITY_ALREADY_RECORDED", code, key)
            return True
        if db.execute("SELECT COUNT(*) FROM events").fetchone()[0] >= self.settings.max_records:
            LOGGER.error("PLATE outbox camera=%s result=STATE_FULL", code)
            return False
        plate = normalize_plate(payload["vehicle"]["license_plate"]["text"])
        kind = str(payload.get("violation_type_code") or "RECOGNITION").upper()
        direction = payload.get("direction") or "IN"
        state = "PENDING"
        # Empty plates must not collapse unrelated vehicles into UNKNOWN_PLATE.
        if plate and db.execute("""SELECT 1 FROM events WHERE camera_id=? AND plate=? AND kind=?
                AND direction=? AND state IN ('PENDING','ACK') AND ABS(occurred-?) < ? LIMIT 1""",
                (event["camera_id"], plate, kind, direction, moment.timestamp(), self.settings.cooldown_seconds)).fetchone():
            state = "SUPPRESSED"
        # A violation PLATE payload already contains recognition data. Coalesce
        # only before handing the recognition to MQTT, never rewrite its time.
        if plate and state == "PENDING":
            if kind == "RECOGNITION":
                if db.execute("""SELECT 1 FROM events WHERE camera_id=? AND plate=? AND kind!='RECOGNITION'
                        AND state IN ('PENDING','ACK') AND ABS(occurred-?)<=1 LIMIT 1""",
                        (event["camera_id"], plate, moment.timestamp())).fetchone():
                    state = "MERGED"
            else:
                candidates = db.execute("""SELECT entity_id FROM events WHERE camera_id=? AND plate=?
                        AND kind='RECOGNITION' AND state='PENDING' AND ABS(occurred-?)<=1""",
                        (event["camera_id"], plate, moment.timestamp())).fetchall()
                for candidate in candidates:
                    if candidate[0] not in self._inflight:
                        db.execute("UPDATE events SET state='MERGED' WHERE entity_id=?", (candidate[0],))
                        self._retry_at.pop(candidate[0], None)
                        LOGGER.info("PLATE camera=%s object_key=%s result=RECOGNITION_MERGED_INTO_VIOLATION",
                                    code, candidate[0])
        if state == "PENDING" and db.execute("SELECT COUNT(*) FROM events WHERE state='PENDING'").fetchone()[0] >= self.settings.max_pending:
            db.rollback()
            LOGGER.warning("PLATE outbox camera=%s result=QUEUE_FULL", code)
            return False
        db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,NULL,?)",
                   (key, event["camera_id"], code, plate, kind, direction, moment.timestamp(), now, state, body))
        db.commit()
        LOGGER.info("PLATE outbox camera=%s object_key=%s kind=%s result=%s", code, key, kind,
                    {"SUPPRESSED": "COOLDOWN_SUPPRESSED", "MERGED": "RECOGNITION_MERGED_INTO_VIOLATION"}.get(state, "PERSISTED"))
        return True

    def _deliver(self, db, now):
        for key in tuple(self._inflight):
            receipt = self.status(key) if self.status else ("ACK", 0.)
            if receipt and receipt[0] == "ACK":
                db.execute("UPDATE events SET state='ACK', ack_at=? WHERE entity_id=?", (now, key))
                db.commit()
                del self._inflight[key]
                LOGGER.info("PLATE outbox object_key=%s result=BROKER_ACK_NOT_BACKEND_CONFIRMATION", key)
            elif receipt and receipt[0] == "REJECTED":
                del self._inflight[key]
                self._retry_at[key] = now+1.
        rows = db.execute("SELECT * FROM events WHERE state='PENDING' ORDER BY kind='RECOGNITION', created, rowid LIMIT ?",
                          (self.settings.max_pending,)).fetchall()
        sent = 0
        for row in rows:
            key = row["entity_id"]
            # Once accepted by Paho, it owns QoS retries. Do not enqueue it again.
            if key in self._inflight:
                continue
            if now-row["created"] > self.settings.ttl_seconds:
                db.execute("UPDATE events SET state='EXPIRED' WHERE entity_id=?", (key,))
                db.commit()
                self._retry_at.pop(key, None)
                LOGGER.error("PLATE outbox camera=%s object_key=%s result=EXPIRED_NOT_DELIVERED", row["camera_code"], key)
                continue
            if now < self._retry_at.get(key, 0.) or sent >= 8:
                continue
            if row["kind"] == "RECOGNITION" and not self.recognition_enabled:
                continue
            if row["kind"] == "RECOGNITION" and now-row["created"] < self.settings.recognition_hold_seconds:
                continue
            if self.eligible is not None and not self.eligible(row["camera_id"], row["camera_code"]):
                continue
            identity = (row["camera_id"], row["plate"])
            if identity in self._inflight.values():
                continue
            ack = db.execute("SELECT MAX(ack_at) FROM events WHERE camera_id=? AND plate=?",
                             identity).fetchone()[0]
            if ack is not None and now-ack < self.guard:
                continue
            if row["kind"] != "RECOGNITION" and db.execute("""SELECT 1 FROM events WHERE camera_id=?
                    AND plate=? AND kind='RECOGNITION' AND state='ACK' AND ABS(occurred-?)<=1 LIMIT 1""",
                    (*identity, row["occurred"])).fetchone():
                LOGGER.error("PLATE camera=%s object_key=%s result=BACKEND_EVENT_DEDUP_REQUIRED", row["camera_code"], key)
            self._retry_at[key] = now+1.
            sent += 1
            try:
                if self.publish(json.loads(row["body"]), camera_code=row["camera_code"], object_key=key):
                    self._inflight[key] = identity
                    self._retry_at.pop(key, None)
            except Exception as error:
                LOGGER.error("PLATE outbox object_key=%s result=PUBLISH_FAILED type=%s", key, type(error).__name__)

    def _run(self):
        db = None
        try:
            db = self._open()
            self._ready.set()
            cleanup_at = 0.
            while True:
                now = time.time()
                if now >= cleanup_at:
                    db.execute("DELETE FROM events WHERE state!='PENDING' AND created<?", (now-self.settings.retention_seconds,))
                    db.commit()
                    cleanup_at = now+30.
                for _ in range(self.settings.max_pending):
                    try:
                        key, code, body = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self._result(key, self._accept(db, key, code, body, now))
                    except (ValueError, TypeError, KeyError, sqlite3.Error) as error:
                        db.rollback()
                        LOGGER.error("PLATE outbox object_key=%s result=PERSIST_FAILED type=%s", key, type(error).__name__)
                        self._result(key, False)
                if self._stop.is_set():
                    break
                self._deliver(db, now)
                self._wake.wait(.1)
                self._wake.clear()
        except Exception as error:
            self._error = error
            LOGGER.error("PLATE outbox result=WORKER_FAILED type=%s", type(error).__name__)
        finally:
            self._ready.set()
            if db is not None:
                db.close()
