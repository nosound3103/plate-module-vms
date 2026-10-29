"""MQTT WebSocket/TLS transport for configuration and PLATE event publishing."""
import json
import logging
import ssl
import threading
import time
from collections import OrderedDict

LOGGER = logging.getLogger(__name__)


def camera_zone_topics(code):
    return [f"smart_vms/cameras/{code}/ai_zones",
            f"smart_vms/cameras/{code}/ai_zones/PLATE/#"]


def create_client(settings, *, client_id):
    """Create a WebSocket TLS client with the service connection policy."""
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                         transport="websockets", reconnect_on_failure=True)
    client.ws_set_options(path=settings.websocket_path)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.username_pw_set(settings.username, settings.password)
    client.reconnect_delay_set(min_delay=1, max_delay=settings.reconnect_max_seconds)
    client.on_connect_fail = lambda *_: LOGGER.warning(
        "MQTT connection attempt failed client_id=%s; reconnect scheduled", client_id)
    return mqtt, client


class MqttTransport:
    def __init__(self, settings):
        self.settings = settings
        self._mqtt, self._client = create_client(settings, client_id=settings.client_id)
        self._client.max_queued_messages_set(128)
        self._client.max_inflight_messages_set(20)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        self._client.on_publish = self._on_publish
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._pending_publishes = {}
        self._publication_status = OrderedDict()
        self._camera_codes = set()
        self._message_handler = lambda *_: None
        self._connect_handler = lambda: None

    def set_handlers(self, on_message, on_connected):
        self._message_handler, self._connect_handler = on_message, on_connected

    def start(self):
        self._client.connect_async(self.settings.broker_host, self.settings.broker_port,
                                   keepalive=self.settings.keepalive_seconds)
        self._client.loop_start()

    def stop(self):
        self._connected.clear()
        self._client.disconnect()
        self._client.loop_stop()

    def _subscribe(self, topics):
        result, _ = self._client.subscribe(topics)
        if result != self._mqtt.MQTT_ERR_SUCCESS:
            LOGGER.warning("MQTT subscription could not be queued result=%s", result)

    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            LOGGER.error("MQTT connection refused result=%s", reason_code)
            return
        with self._lock:
            codes = tuple(self._camera_codes)
        topics = [(f"smart_vms/cameras/zone/{z}", 1) for z in self.settings.zone_ids]
        topics.extend((topic, 1) for code in codes for topic in camera_zone_topics(code))
        self._subscribe(topics)
        self._connected.set()
        # The service receives this reset before any retained configuration.
        self._connect_handler()
        LOGGER.info("MQTT connected client_id=%s subscriptions=%d", self.settings.client_id, len(topics))

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _properties):
        self._connected.clear()
        LOGGER.warning("MQTT disconnected result=%s; reconnect scheduled", reason_code)

    def _on_subscribe(self, _client, _userdata, _mid, reason_codes, _properties):
        failures = sum(code.is_failure for code in reason_codes)
        if failures:
            LOGGER.error("MQTT subscription rejected count=%d; check broker ACL", failures)
        else:
            LOGGER.info("MQTT subscriptions acknowledged count=%d", len(reason_codes))

    def _on_message(self, _client, _userdata, message):
        if len(message.payload) > self.settings.config_max_bytes:
            LOGGER.warning("Configuration message rejected result=PAYLOAD_TOO_LARGE")
            return
        self._message_handler(message.topic, bytes(message.payload))

    def _on_publish(self, _client, _userdata, mid, reason_code, _properties):
        with self._publish_lock:
            context = self._pending_publishes.pop(mid, None)
            if context is not None:
                self._publication_status[context[1]] = (
                    "REJECTED" if getattr(reason_code, "is_failure", False) else "ACK", time.monotonic())
                while len(self._publication_status) > 1024:
                    self._publication_status.popitem(last=False)
        if context is not None:
            camera_code, object_key = context
            status = "BROKER_REJECTED" if getattr(reason_code, "is_failure", False) else "BROKER_ACK"
            LOGGER.info("publish topic=smart_vms/ai_events/PLATE camera=%s object_key=%s result=%s mid=%d",
                        camera_code, object_key, status, mid)

    def publish_plate_event(self, event, *, camera_code, object_key):
        """Return queue acceptance; PUBACK is logged asynchronously, never awaited."""
        from plate_ai.messaging.events import PLATE_TOPIC
        try:
            body = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (ValueError, TypeError):
            LOGGER.error("publish topic=%s camera=%s object_key=%s result=INVALID_PAYLOAD",
                         PLATE_TOPIC, camera_code, object_key)
            return False
        with self._publish_lock:
            if not self._connected.is_set():
                LOGGER.warning("publish topic=%s camera=%s object_key=%s result=DISCONNECTED",
                               PLATE_TOPIC, camera_code, object_key)
                return False
            if len(self._pending_publishes) >= 128:
                LOGGER.warning("publish topic=%s camera=%s object_key=%s result=QUEUE_FULL",
                               PLATE_TOPIC, camera_code, object_key)
                return False
            try:
                info = self._client.publish(PLATE_TOPIC, body, qos=1, retain=False)
            except Exception as error:
                LOGGER.error("publish topic=%s camera=%s object_key=%s result=FAILED type=%s",
                             PLATE_TOPIC, camera_code, object_key, type(error).__name__)
                return False
            # QoS 1 messages can remain in Paho's outgoing queue if disconnect
            # races publish(). Retrying these would create another application message.
            accepted = info.rc in {self._mqtt.MQTT_ERR_SUCCESS, self._mqtt.MQTT_ERR_NO_CONN}
            if accepted:
                self._pending_publishes[info.mid] = (camera_code, object_key)
                self._publication_status[object_key] = ("PENDING", time.monotonic())
                while len(self._publication_status) > 1024:
                    self._publication_status.popitem(last=False)
            status = ("QUEUED" if info.rc == self._mqtt.MQTT_ERR_SUCCESS else "QUEUED_RECONNECT") if accepted else f"REJECTED_RC_{info.rc}"
            LOGGER.info("publish topic=%s camera=%s object_key=%s result=%s mid=%d",
                        PLATE_TOPIC, camera_code, object_key, status, info.mid)
            return accepted

    def publication_status(self, object_key):
        """Broker receipt time for coordinating the backend's plate dedup window."""
        with self._publish_lock:
            if any(key == object_key for _, key in self._pending_publishes.values()):
                return ("PENDING", 0.)
            return self._publication_status.get(object_key)

    def subscribe_camera(self, code):
        with self._lock:
            self._camera_codes.add(code)
        if self._connected.is_set():
            self._subscribe([(topic, 1) for topic in camera_zone_topics(code)])

    def unsubscribe_camera(self, code):
        with self._lock:
            self._camera_codes.discard(code)
        if self._connected.is_set():
            self._client.unsubscribe(camera_zone_topics(code))

    def request_cameras(self):
        if not self._connected.is_set():
            return
        topic = "smart_vms/cameras/getdevice"
        for zone_id in self.settings.zone_ids:
            data = {"zone_id": zone_id, "organization_id": self.settings.organization_id}
            result = self._client.publish(topic, json.dumps(data), qos=1, retain=False)
            status = "QUEUED" if result.rc == self._mqtt.MQTT_ERR_SUCCESS else f"REJECTED_RC_{result.rc}"
            LOGGER.info("publish topic=%s camera=- object_key=zone:%s result=%s", topic, zone_id, status)
