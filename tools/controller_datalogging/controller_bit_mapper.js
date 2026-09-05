const HID = require('node-hid');

const path = "\\\\?\\HID#VID_054C&PID_09CC&MI_03#8&39aa2fc4&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}";

const device = new HID.HID(path);

device.on('data', (data) => {
  console.log(data);
});

device.on('error', (err) => {
  console.error('Error:', err);
});

console.log('Listening the controller... Move the joysticks and look at the shell.');