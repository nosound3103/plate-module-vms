"""Dedicated MQTT connection for disposable realtime overlays."""
import json
import logging
import threading
from .mqtt import create_client

LOGGER = logging.getLogger(__name__)


class BboxMqttPublisher:
    def __init__(self, settings):
        self.settings = settings
        self._mqtt, self._client = create_client(settings, client_id=settings.bbox_client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._inflight = None

    def start(self):
        self._client.connect_async(self.settings.broker_host, self.settings.broker_port,
                                   keepalive=self.settings.keepalive_seconds)
        self._client.loop_start()

    def stop(self):
        self._connected.clear()
        self._client.disconnect()
        self._client.loop_stop()

    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            LOGGER.error("MQTT bbox connection refused result=%s", reason_code)
            return
        self._connected.set()
        LOGGER.info("MQTT bbox connected")

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _properties):
        self._connected.clear()
        with self._lock:
            self._inflight = None
        LOGGER.warning("MQTT bbox disconnected result=%s; reconnect scheduled", reason_code)

    def publish(self, message, *, topic_prefix="smart_vms/ai/bbox"):
        """Drop under backpressure; allow one unsent overlay packet per connection."""
        if topic_prefix not in {"smart_vms/ai/bbox", "smart_vms/bbox"}:
            raise ValueError("Invalid bbox topic prefix")
        code = message["camera_code"]
        topic, object_key = f"{topic_prefix}/{code}", f"frame:{message['frame_sequence']}"
        try:
            body = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (ValueError, TypeError):
            LOGGER.error("publish topic=%s camera=%s object_key=%s result=INVALID_PAYLOAD", topic, code, object_key)
            return False
        with self._lock:
            if not self._connected.is_set():
                status, accepted = "DROPPED_DISCONNECTED", False
            elif self._inflight is not None and not self._inflight.is_published():
                status, accepted = "DROPPED_BUSY", False
            else:
                try:
                    info = self._client.publish(topic, body, qos=0, retain=False)
                    accepted = info.rc == self._mqtt.MQTT_ERR_SUCCESS
                    self._inflight = info if accepted else None
                    status = "QUEUED_QOS0" if accepted else f"DROPPED_RC_{info.rc}"
                except Exception as error:
                    status, accepted = f"DROPPED_{type(error).__name__}", False
            LOGGER.info("publish topic=%s camera=%s object_key=%s result=%s", topic, code, object_key, status)
            return accepted
