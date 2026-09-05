#include <IRremote.hpp>

#define IR_SEND_PIN 15

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
  IrSender.begin(IR_SEND_PIN);
}

void loop() {
  uint8_t yaw = 63;
  uint8_t pitch = 63;
  uint8_t throttle = 0;
  uint8_t trim = 63;

  unsigned long currentMillis = millis();

  // 1. FASE D'ARMAMENT (Primers 6 segons, innegociable per seguretat)
  if (currentMillis < 2000) {
    throttle = 0;   
  } 
  else if (currentMillis < 4000) {
    throttle = 40; 
  } 
  else if (currentMillis < 6000) {
    throttle = 0;   
  } 
  
  // 2. EL TEU BUCLE DE 3 SEGONS
  else {
    // Rellotge intern cíclic de 3000 mil·lisegons
    unsigned long loopTime = (currentMillis - 6000) % 6000; 

    if (loopTime <= 1500) {
      // BUCLE 1 (0 a 1 segons): Hèlixs generals
      throttle = 30; 
      pitch = 63;
      yaw = 63;
    } 
    if (1500 < loopTime && loopTime <= 3000) {
      // BUCLE 2 (1 a 3 segons): Hèlix del darrere "únicament"
      // Hi deixem el gas a 20 perquè la placa habiliti la cua
      throttle = 0; 
      pitch = 63; // Màxim cap endavant
      yaw = 63;
    } 
    if (3000 < loopTime && loopTime <= 4500) {
      // BUCLE 2 (1 a 3 segons): Hèlix del darrere "únicament"
      // Hi deixem el gas a 20 perquè la placa habiliti la cua
      throttle = 20; 
      pitch = 0; // Màxim cap endavant
      yaw = 63;
    }
    if (loopTime> 4500){
      throttle = 0;
      pitch = 63;
      yaw = 63;
    }
  }

  sendSyma107G(yaw, pitch, throttle, trim, 0);
  delay(20);
}