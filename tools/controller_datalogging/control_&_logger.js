const HID = require('node-hid');
const { SerialPort } = require('serialport');
const fs = require('fs');

const clamp = (x, a, b) => Math.max(Math.min(x, b), a);

// Controller path 
const CONTROLLER_PATH = "\\\\?\\HID#VID_054C&PID_09CC&MI_03#8&302cc5d&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}";
const SERIAL_PATH = "COM5";

const state = [0xFF, 0xAA, 0, 63, 63, 63]; // header, Throttle, Yaw, Pitch, Trim

// --- GLOBAL ADJUSTMENT VARIABLES ---
let internalThrottle = 0; 
// As larger the number, less sensibility has the right joystick (PITCH & YAW)
const sensibility_R = 2.5; 

const LOG_FILE = "signals.csv";
const logStream = fs.createWriteStream(LOG_FILE, { flags: 'w' });
logStream.write("TEMPS;THROTTLE;YAW;PITCH;TRIM\n"); // Headtitle
const startTime = Date.now();

const controller = new HID.HID(CONTROLLER_PATH);
const serialPort = new SerialPort({ path: SERIAL_PATH, baudRate: 9600, autoOpen: false });

serialPort.open(err => {
  if (err) {
    console.error('Error opening the serial port:', err.message);
    return;
  }
  console.log('Serisl Port Opened. Listening the controller...');

  controller.on('data', data => {
    const leftY = data[2];
    const rightX = data[3];
    const rightY = data[4];

    // Accumulate the value, /200 to adjust the sensibility
    internalThrottle = clamp(internalThrottle + (128 - leftY) / 200, 0, 127);
    state[2] = Math.round(internalThrottle); 

    // Yaw: invert the signe
    let calculYaw = 63 - ((rightX - 128) / sensibility_R);
    state[3] = Math.round(clamp(calculYaw, 0, 127));

    // Pitch: calculated form the middle
    let calculPitch = 63 + ((rightY - 128) / sensibility_R);
    state[4] = Math.round(clamp(calculPitch, 0, 127));
  });

  // Independent sending loop (every 40 ms)
  setInterval(() => {
    serialPort.write(Buffer.from(state));
  }, 40);

  setInterval(() => {
    const elapsedMs = Date.now() - startTime;
    // state[2]=throttle, state[3]=yaw, state[4]=pitch, state[5]=trim
    logStream.write(`${elapsedMs};${state[2]};${state[3]};${state[4]};${state[5]}\n`);
  }, 40);

  controller.on('error', err => console.error('Controller Error', err));
});