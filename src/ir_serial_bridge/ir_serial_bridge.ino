#include <IRremote.hpp>

#define IR_SEND_PIN 15

uint8_t Throttle = 0;  // 0-127
uint8_t Yaw = 63;      // 0-127, centre=63
uint8_t Pitch = 63;    // 0-127, centre=63
uint8_t Trim = 63;     // 0-127, centre=63
uint8_t state = 0;

void sendSyma107G(uint8_t yaw, uint8_t pitch, uint8_t throttle, uint8_t trim, uint8_t channel) {
  IrSender.enableIROut(38);
  IrSender.mark(2000);
  IrSender.space(2000);

  uint8_t data[4];
  data[3] = yaw;
  data[2] = pitch;
  data[1] = throttle | (channel << 7);
  data[0] = trim;

  for (int8_t i = 31; i >= 0; i--) {
    IrSender.mark(300);
    if ((data[i / 8] >> (i % 8)) & 1) {
      IrSender.space(700);
    } else {
      IrSender.space(300);
    }
  }
}

void setup() {
  Serial.begin(9600);
  IrSender.begin(IR_SEND_PIN);
}

void loop() {
  while (Serial.available() > 0) {
    uint8_t r = (uint8_t) Serial.read();
    switch (state) {
      case 0: if (r == 0xFF) state++; break;
      case 1: if (r == 0xAA) state++; else state = 0; break;
      case 2: Throttle = r; state++; break;
      case 3: Yaw = r; state++; break;
      case 4: Pitch = r; state++; break;
      case 5: Trim = r; state = 0; break;
    }
  }
  sendSyma107G(Yaw, Pitch, Throttle, Trim, 0);
  delay(20);
}