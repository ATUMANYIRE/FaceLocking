#include <Servo.h>

Servo panServo;
const int SERVO_PIN = 2;  // GPIO2, silkscreen label "D4" on most ESP8266 boards

void setup() {
  Serial.begin(9600);
  panServo.attach(SERVO_PIN);
  panServo.write(90);
}

void loop() {
  if (Serial.available() > 0) {
    int angle = Serial.parseInt();
    if (angle >= 0 && angle <= 180) {
      panServo.write(angle);
    }
  }
}