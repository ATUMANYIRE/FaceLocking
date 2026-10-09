# MQTT face tracking (ESP8266 + 28BYJ-48 stepper)

The vision script on the PC is a Paho MQTT **publisher**. The ESP8266 **subscribes** and turns the stepper.

```
camera → recognize + lock (PC) ──publish──▶ Mosquitto broker ──▶ ESP8266 ──▶ ULN2003 ──▶ 28BYJ-48
                                face_tracking/angle   "0".."180"
```

## Wiring

| ULN2003 | ESP8266 (NodeMCU / D1 mini) |
|---------|-----------------------------|
| IN1     | D1 (GPIO5)  |
| IN2     | D2 (GPIO4)  |
| IN3     | D5 (GPIO14) |
| IN4     | D6 (GPIO12) |
| +  (5–12V) | 5V (VIN / VU, or an external 5V supply) |
| −       | GND (must be shared with the ESP8266) |

Don't power the motor from the 3.3V pin.

## Topics

| Topic | Direction | Payload |
|-------|-----------|---------|
| `face_tracking/angle`    | PC → ESP | target pan angle `0`–`180` (90 = centre) |
| `face_tracking/cmd`      | PC → ESP | `home` (current position becomes 90°), `release` (coils off) |
| `face_tracking/status`   | ESP → PC | `online` / `offline` (retained, last-will) |
| `face_tracking/position` | ESP → PC | angle reached after each move |

The stepper has no position sensor, so **the position it has at power-on is treated as 90°**. Centre the camera by hand before powering up. If it drifts, press `r` (release the coils), turn it back to centre by hand, then press `h`.

## 1. Broker (Mosquitto on the PC)

Mosquitto 2.x only accepts connections from localhost by default, so the ESP8266 can't reach it. Either stop the Windows service and run this folder's config (in an admin terminal):

```powershell
net stop mosquitto
& "C:\Program Files\mosquitto\mosquitto.exe" -c mqtt_tracking\mosquitto.conf -v
```

or add the two lines from `mosquitto.conf` to `C:\Program Files\mosquitto\mosquitto.conf` and restart the service. Allow inbound TCP 1883 in Windows Firewall. Get the PC's LAN IP from `ipconfig`.

## 2. ESP8266

1. Arduino IDE → Library Manager → install **PubSubClient** (Nick O'Leary).
2. Copy `esp8266_stepper_mqtt/secrets.example.h` to `secrets.h` (git-ignored) and set `WIFI_SSID`, `WIFI_PASSWORD`, and `MQTT_HOST` (the PC's LAN IP). The PC and the ESP8266 must be on the same network.
3. Board: *NodeMCU 1.0 (ESP-12E Module)* (or your board), then upload.
4. Serial Monitor at 115200 should show `[wifi] connected` and `[mqtt] connecting ... ok`. You can also type an angle such as `45` there to test the motor without the PC.

Tuning (top of the sketch): `STEP_INTERVAL_US` sets the speed (it stalls below about 1000 µs), and `IDLE_RELEASE_MS` sets how long the coils stay powered after a move.

## 3. PC (from the repo root)

```bash
pip install paho-mqtt
python -m mqtt_tracking.mqtt_publisher                 # no camera: sends 90 → 60 → 120 → 90
python -m mqtt_tracking.face_track_mqtt                # broker on this PC
python -m mqtt_tracking.face_track_mqtt --invert       # if the motor turns away from the face
python -m mqtt_tracking.face_track_mqtt --mode onboard # camera mounted ON the motor
python -m mqtt_tracking.face_track_mqtt --width 1920 --height 1080   # camera resolution (default 1280x720)
```

To watch the traffic: `mosquitto_sub -t "face_tracking/#" -v`

Keys: `q` quit · `f` fullscreen · `i` invert needle · `l` release lock · `c` centre · `h` home (here = 90°) · `r` release coils · `+/-` threshold.

`--mode fixed` (the default) is for a stationary camera such as the laptop webcam: the face position maps straight to the motor angle. `--mode onboard` assumes the camera rides on the motor. Each correction moves the motor toward the face until the face is centred. Tune `TRACK_GAIN_DEG`, `TRACK_UPDATE_S`, and `TRACK_DEADBAND` in `face_track_mqtt.py` if it overshoots or reacts too slowly.
