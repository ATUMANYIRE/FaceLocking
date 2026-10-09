// esp8266_stepper_mqtt_vertical.ino
// Vertical-only build: drives ONE 28BYJ-48 (tilt) through a ULN2003 from MQTT.
// Pairs with mqtt_tracking/face_track_mqtt_vertical.py.
//
// Wiring (ULN2003 -> ESP8266):  IN1=D1  IN2=D2  IN3=D5  IN4=D6
// Power the ULN2003 board from 5V (VIN/USB 5V or external), NOT 3.3V. Common GND.
// Stepper plugs into the ULN2003 board's white 5-pin JST socket (keyed; only one way).
//
// Topics (base = "face_tracking/tilt"):
//   face_tracking/tilt/angle     <- "0".."180"  target tilt angle (90 = centre)
//   face_tracking/tilt/cmd       <- "home"      current physical position becomes 90 deg
//                                   "release"   de-energise coils (turn by hand freely)
//   face_tracking/tilt/status    -> "online" / "offline" (retained, LWT)
//   face_tracking/tilt/position  -> angle reached after a move
//
// Libraries: ESP8266 board package + "PubSubClient" by Nick O'Leary (Library Manager).
// Serial Monitor (115200): type an angle (e.g. 45) or "home"/"release" to test the motor.

#include <ESP8266WiFi.h>
#include <PubSubClient.h>

#include "secrets.h"
const uint16_t MQTT_PORT = 1883;

const char* BASE_TOPIC = "face_tracking/tilt";

const uint8_t COIL_PINS[4] = {D1, D2, D5, D6};   // IN1..IN4

const float STEPS_PER_REV = 4096.0;
const float STEPS_PER_DEG = STEPS_PER_REV / 360.0;
const unsigned long STEP_INTERVAL_US = 1500;
const unsigned long IDLE_RELEASE_MS  = 2000;
const int ANGLE_MIN = 0;
const int ANGLE_MAX = 180;

const uint8_t HALF_STEP[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1},
};

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

String topicAngle, topicCmd, topicStatus, topicPosition;

long currentStep = 0;
long targetStep  = 0;
uint8_t phase = 0;
bool coilsOn = false;
bool positionReported = true;
unsigned long lastStepUs = 0;
unsigned long lastMoveMs = 0;
unsigned long lastMqttAttemptMs = 0;

long angleToStep(float deg) { return lround(deg * STEPS_PER_DEG); }
float stepToAngle(long s)   { return s / STEPS_PER_DEG; }

void writeCoils(uint8_t p) {
  for (int i = 0; i < 4; i++) digitalWrite(COIL_PINS[i], HALF_STEP[p][i]);
  coilsOn = true;
}

void releaseCoils() {
  for (int i = 0; i < 4; i++) digitalWrite(COIL_PINS[i], LOW);
  coilsOn = false;
}

void setTargetAngle(float deg) {
  if (deg < ANGLE_MIN) deg = ANGLE_MIN;
  if (deg > ANGLE_MAX) deg = ANGLE_MAX;
  targetStep = angleToStep(deg);
}

void runStepper() {
  unsigned long nowUs = micros();
  if (currentStep == targetStep) {
    if (!positionReported) {
      positionReported = true;
      Serial.printf("[move] reached %d deg\n", (int)lround(stepToAngle(currentStep)));
      if (mqtt.connected()) mqtt.publish(topicPosition.c_str(), String((int)lround(stepToAngle(currentStep))).c_str());
    }
    if (coilsOn && millis() - lastMoveMs > IDLE_RELEASE_MS) releaseCoils();
    return;
  }
  if (nowUs - lastStepUs < STEP_INTERVAL_US) return;
  lastStepUs = nowUs;

  if (targetStep > currentStep) { currentStep++; phase = (phase + 1) & 7; }
  else                          { currentStep--; phase = (phase + 7) & 7; }
  writeCoils(phase);
  lastMoveMs = millis();
  positionReported = false;
}

void handleText(const char* topic, const String& msg) {
  if (topicCmd == topic) {
    if (msg == "home") {
      currentStep = targetStep = angleToStep(90);
      Serial.println("[cmd] home: current position = 90 deg");
    } else if (msg == "release") {
      targetStep = currentStep;
      releaseCoils();
      Serial.println("[cmd] coils released");
    }
    return;
  }
  if (msg.length() == 0 || !(isDigit(msg[0]) || msg[0] == '-')) return;
  setTargetAngle(msg.toFloat());
  if (targetStep == currentStep) Serial.printf("[move] already at %d deg\n", (int)lround(stepToAngle(currentStep)));
  else Serial.printf("[move] -> %d deg (%ld steps)\n", (int)lround(stepToAngle(targetStep)), targetStep - currentStep);
}

void onMqttMessage(char* topic, byte* payload, unsigned int len) {
  String msg;
  msg.reserve(len);
  for (unsigned int i = 0; i < len; i++) msg += (char)payload[i];
  msg.trim();
  handleText(topic, msg);
}

void ensureMqtt() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;
  if (millis() - lastMqttAttemptMs < 3000) return;
  lastMqttAttemptMs = millis();

  String clientId = "esp8266-stepper-tilt-" + String(ESP.getChipId(), HEX);
  Serial.printf("[mqtt] connecting to %s:%u ... ", MQTT_HOST, MQTT_PORT);
  if (mqtt.connect(clientId.c_str(), topicStatus.c_str(), 1, true, "offline")) {
    Serial.println("ok");
    mqtt.publish(topicStatus.c_str(), "online", true);
    mqtt.subscribe(topicAngle.c_str());
    mqtt.subscribe(topicCmd.c_str(), 1);
  } else {
    Serial.printf("failed, state=%d\n", mqtt.state());
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) pinMode(COIL_PINS[i], OUTPUT);
  releaseCoils();

  // No position sensor: whatever position the motor has at power-on is taken as 90 deg (centre).
  currentStep = targetStep = angleToStep(90);

  topicAngle    = String(BASE_TOPIC) + "/angle";
  topicCmd      = String(BASE_TOPIC) + "/cmd";
  topicStatus   = String(BASE_TOPIC) + "/status";
  topicPosition = String(BASE_TOPIC) + "/position";

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("\n[wifi] connecting to %s\n", WIFI_SSID);

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
  mqtt.setSocketTimeout(2);
  mqtt.setKeepAlive(15);
}

void loop() {
  static bool wifiWasUp = false;
  bool wifiUp = WiFi.status() == WL_CONNECTED;
  if (wifiUp && !wifiWasUp) Serial.printf("[wifi] connected, IP %s\n", WiFi.localIP().toString().c_str());
  wifiWasUp = wifiUp;

  ensureMqtt();
  mqtt.loop();
  runStepper();

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    handleText((line == "home" || line == "release") ? topicCmd.c_str() : topicAngle.c_str(), line);
  }
}
