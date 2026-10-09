# mqtt_tracking/mqtt_publisher.py
"""
Paho MQTT client that publishes pan angles for the ESP8266 stepper.

Same interface as src/servo_control.ServoController (set_angle / close), but
instead of writing to a serial port it publishes to the broker:

    <base>/angle   "<0-180>"            target pan angle, plain ASCII integer
    <base>/cmd     "home" | "release"   one-shot commands for the ESP8266
    <base>/status  (subscribed)         "online"/"offline" published by the ESP8266

The client connects asynchronously and auto-reconnects in a background
thread, so the vision loop never blocks on the network.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

import paho.mqtt.client as mqtt


class MqttAngleController:
    def __init__(self, host: str = "127.0.0.1", port: int = 1883, base_topic: str = "face_tracking"):
        self.base = base_topic.rstrip("/")
        self.topic_angle = f"{self.base}/angle"
        self.topic_cmd = f"{self.base}/cmd"
        self.topic_status = f"{self.base}/status"

        self.connected = False
        self.device_status = "unknown"   # last value the ESP8266 published on <base>/status
        self.device_position: Optional[int] = None
        self.last_angle: Optional[int] = None

        client_id = f"face-vision-{uuid.uuid4().hex[:6]}"
        try:  # paho-mqtt >= 2.0
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        except AttributeError:  # paho-mqtt 1.x
            self.client = mqtt.Client(client_id=client_id)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)

        print(f"[mqtt] connecting to {host}:{port} (topics under '{self.base}/')")
        self.client.connect_async(host, port, keepalive=30)
        self.client.loop_start()

    # callbacks take *args so they work with both paho 1.x and 2.x signatures
    def _on_connect(self, client, userdata, flags, rc, *args):
        ok = (rc == 0) if isinstance(rc, int) else not rc.is_failure
        self.connected = ok
        if ok:
            print("[mqtt] connected")
            client.subscribe(self.topic_status)
            client.subscribe(f"{self.base}/position")
            self.last_angle = None  # force the next angle to be re-sent after a reconnect
        else:
            print(f"[mqtt] connect refused: {rc}")

    def _on_disconnect(self, client, userdata, *args):
        self.connected = False
        print("[mqtt] disconnected (auto-reconnecting)")

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode(errors="replace").strip()
        if msg.topic == self.topic_status:
            self.device_status = payload
            print(f"[mqtt] device status: {payload}")
        elif msg.topic.endswith("/position"):
            try:
                self.device_position = int(float(payload))
            except ValueError:
                pass

    def set_angle(self, angle: float):
        angle = max(0, min(180, int(round(angle))))
        if angle == self.last_angle:
            return
        if not self.connected:
            return
        self.client.publish(self.topic_angle, str(angle), qos=0)
        self.last_angle = angle

    def send_cmd(self, cmd: str):
        if self.connected:
            self.client.publish(self.topic_cmd, cmd, qos=1)
            print(f"[mqtt] cmd -> {cmd}")

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    # quick manual test without the camera: python -m mqtt_tracking.mqtt_publisher
    ctl = MqttAngleController()
    time.sleep(1.5)
    for a in (90, 60, 120, 90):
        print(f"angle {a}")
        ctl.set_angle(a)
        time.sleep(2)
    ctl.close()
