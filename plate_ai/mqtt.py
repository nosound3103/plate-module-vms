"""MQTT WebSocket/TLS transport for camera discovery and zone synchronization."""
import json
import logging
import ssl
import threading

LOGGER = logging.getLogger(__name__)


class MqttTransport:
    def __init__(self, settings):
        import paho.mqtt.client as mqtt
        self.settings = settings
        self._mqtt = mqtt
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                   client_id=settings.client_id, transport="websockets")
        self._client.ws_set_options(path=settings.websocket_path)
        self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self._client.username_pw_set(settings.username, settings.password)
        self._client.reconnect_delay_set(min_delay=1, max_delay=settings.reconnect_max_seconds)
        self._client.max_queued_messages_set(128)
        self._client.max_inflight_messages_set(20)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        self._connected = threading.Event()
        self._lock = threading.Lock()
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
        topics.extend((f"smart_vms/cameras/{code}/ai_zones/#", 1) for code in codes)
        self._subscribe(topics)
        self._connected.set()
        # The service receives this reset before any retained configuration.
        self._connect_handler()
        LOGGER.info("MQTT connected subscriptions=%d", len(topics))

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _properties):
        self._connected.clear()
        LOGGER.warning("MQTT disconnected result=%s; reconnect scheduled", reason_code)

    def _on_subscribe(self, _client, _userdata, _mid, reason_codes, _properties):
        failures = sum(code.is_failure for code in reason_codes)
        if failures:
            LOGGER.error("MQTT subscription rejected count=%d; check broker ACL", failures)

    def _on_message(self, _client, _userdata, message):
        if len(message.payload) > self.settings.config_max_bytes:
            LOGGER.warning("Configuration message rejected result=PAYLOAD_TOO_LARGE")
            return
        self._message_handler(message.topic, bytes(message.payload))

    def subscribe_camera(self, code):
        with self._lock:
            self._camera_codes.add(code)
        if self._connected.is_set():
            self._subscribe([(f"smart_vms/cameras/{code}/ai_zones/#", 1)])

    def unsubscribe_camera(self, code):
        with self._lock:
            self._camera_codes.discard(code)
        if self._connected.is_set():
            self._client.unsubscribe(f"smart_vms/cameras/{code}/ai_zones/#")

    def request_cameras(self):
        if not self._connected.is_set():
            return
        topic = "smart_vms/cameras/getdevice"
        for zone_id in self.settings.zone_ids:
            data = {"zone_id": zone_id, "organization_id": self.settings.organization_id}
            result = self._client.publish(topic, json.dumps(data), qos=1, retain=False)
            status = "QUEUED" if result.rc == self._mqtt.MQTT_ERR_SUCCESS else f"REJECTED_RC_{result.rc}"
            LOGGER.info("publish topic=%s camera=- object_key=zone:%s result=%s", topic, zone_id, status)
