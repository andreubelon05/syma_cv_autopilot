"""
Autonomous Precision Takeoff and Landing System for IR-Actuated UAV
Author: Andreu Belón Vallés

This script implements a dual-camera computer vision architecture using a YOLO-based 
ONNX model to control an infrared (IR) actuated helicopter in real-time. It features 
a finite state machine (Takeoff, Hover, Descent, Flare, Landed), closed-loop PID 
altitude control, and odometry-based Yaw pulse control with memory recovery.
"""

import cv2
import numpy as np
import sys
import time
import threading
import onnxruntime as ort
import csv
import serial

# ==========================================
# CONFIGURATION
# ==========================================

SERIAL_PORT = "COM5"      
BAUD_RATE = 9600
SERIAL_INTERVAL = 0.040   # 40 ms between serial packets (25 Hz)

# --- Global & Security Parameters ---
CONFIRMATION_FRAMES = 2    
CAM_RES_Y = 480            

# --- Camera Transition Logic ---
CAM_TRANSITION_FRAMES = 5  
MIN_TRANSITION_SEC = 2.75  

# --- State Machine Thresholds (Ratio & Pixel Heights) ---
TARGET_RATIO_ALT_CAM0 = 2.9  
TARGET_RATIO_ALT_CAM1 = 5    
CY_TRIGGER_DESCENT = 180   

RATIO_TRIGGER_FLARE = 3.05  
FLARE_DELAY_SEC = 0.5    

# --- Emergency Proximity Kill-Switch ---
CY_GROUND_EMERGENCY = 355
HEIGHT_GROUND_EMERGENCY = 110

# --- Throttle Power Profiles ---
THROTTLE_START = 40
THROTTLE_HOVER_BASE = 70
THROTTLE_TAKEOFF = THROTTLE_HOVER_BASE - 6   
TAKEOFF_RAMP_TIME = 1.5        
KP_ALT = 3.25

THROTTLE_DESCENT_START = THROTTLE_HOVER_BASE - 7
THROTTLE_DESCENT_END = THROTTLE_HOVER_BASE - 11   
DESCENT_RAMP_TIME = 0.65      
DESCENT_TIMEOUT_SEC = 1.5    
DESCENT_MIN_SEC = 0.25        

THROTTLE_FLARE = THROTTLE_HOVER_BASE + 2
HOVER_STABILIZE_CAM1_SEC = 6.0  

# --- Odometry-based Yaw Pulse Control ---
TARGET_X_CAM0 = 320       
TARGET_X_CAM1 = 320       
YAW_DEADZONE_R = 45         
YAW_DEADZONE_L = 20
YAW_TURN_POWER = 20       
YAW_K_TIME = 0.004        
YAW_MAX_TIME = 0.25        
YAW_COOLDOWN = 0.3       
YAW_MAX_MEMORY = 0.85
YAW_RECOVERY_FACTOR = 0.7  

PITCH_NEUTRAL = 60
YAW_NEUTRAL = 65
TRIM_NEUTRAL = 100

MODEL_PATH = "model_syma.onnx"          
CAMERA_INDICES = [0, 1]              
CONFIDENCE_THRESHOLD = 0.40          
OUTPUT_VIDEO_TEMPLATE = "cam_0{idx}.mp4"
MAX_TRAJECTORY_LEN = 15             
DEBUG_PRINT_EVERY_N_FRAMES = 0      

PREFERRED_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
CAMERA_WARMUP_ATTEMPTS = 8
CAMERA_WARMUP_DELAY = 0.15

CAMERA_MANUAL_SETTINGS = {
    1: {  
        "autofocus": False,
        "focus": 0,
        "auto_exposure": False,   
        "exposure": -7,
        "zoom": 0,
        "pan": 0,
        "tilt": 0,
    },
}
# ==========================================

def apply_manual_settings(cap, settings):
    """
    Applies manual hardware settings (exposure, focus, etc.) to the camera 
    using OpenCV properties, bypassing OS automatic adjustments.
    """
    if not settings:
        return {}

    prop_map = {
        "autofocus": cv2.CAP_PROP_AUTOFOCUS,
        "focus": cv2.CAP_PROP_FOCUS,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "zoom": cv2.CAP_PROP_ZOOM,
        "pan": cv2.CAP_PROP_PAN,
        "tilt": cv2.CAP_PROP_TILT,
    }

    if "autofocus" in settings:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if settings["autofocus"] else 0)
    if "auto_exposure" in settings:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if settings["auto_exposure"] else 0.25)
    if "focus" in settings:
        cap.set(cv2.CAP_PROP_FOCUS, settings["focus"])
    if "exposure" in settings:
        cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    if "zoom" in settings:
        cap.set(cv2.CAP_PROP_ZOOM, settings["zoom"])
    if "pan" in settings:
        cap.set(cv2.CAP_PROP_PAN, settings["pan"])
    if "tilt" in settings:
        cap.set(cv2.CAP_PROP_TILT, settings["tilt"])

    applied = {}
    for name in settings:
        prop = prop_map.get(name)
        if prop is not None:
            applied[name] = cap.get(prop)
    return applied


class CameraWorker(threading.Thread):
    """
    Parallel thread for real-time video capture, image preprocessing, 
    and ONNX machine learning inference. Operates independently to avoid 
    blocking the main flight control loop.
    """
    def __init__(self, index, stop_event):
        super().__init__(daemon=True)
        self.index = index
        self.stop_event = stop_event

        options = ort.SessionOptions()
        options.intra_op_num_threads = 6 
        
        self.session = ort.InferenceSession(MODEL_PATH, options)
        self.input_name = self.session.get_inputs()[0].name
        
        shape = self.session.get_inputs()[0].shape
        self.input_h, self.input_w = shape[1], shape[2] 
        self.is_floating_model = True 
        self.input_scale, self.input_zero_point = 0.0, 0

        self.cap = cv2.VideoCapture(index, PREFERRED_BACKEND)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(index) 

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera {index}")

        warm_ok = False
        for _ in range(CAMERA_WARMUP_ATTEMPTS):
            ret, _ = self.cap.read()
            if ret:
                warm_ok = True
                break
            time.sleep(CAMERA_WARMUP_DELAY)
        if not warm_ok:
            self.cap.release()
            raise RuntimeError(f"Camera {index} initialized but capturing empty frames.")

        settings = CAMERA_MANUAL_SETTINGS.get(index)
        if settings:
            apply_manual_settings(self.cap, settings)

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and not np.isnan(fps) and fps > 0 else 30.0

        self.out_path = OUTPUT_VIDEO_TEMPLATE.format(idx=index)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (self.frame_w, self.frame_h))
        
        self.csv_filename = f"boxes_0{index}.csv"
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter=';')
        self.csv_writer.writerow(["time_ms", "cx", "cy", "width", "height", "ratio"])
        self.start_time = time.time()
        
        self.window_name = f"Flight Test - Camera {index}"
        self.trajectory = []
        self.traj_lock = threading.Lock()
        self.total_tracked_points = 0
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_height = 0  
        self.latest_cx = 0
        self.latest_cy = 0
        
        self.frame_count = 0
        self.last_infer_ms = 0.0

        print(f"[INFO] Camera {index}: {self.frame_w}x{self.frame_h} @ {self.fps:.1f} FPS")

        self._record_start_time = None
        self._next_frame_slot = 0

    def preprocess(self, frame_bgr):
        """
        Center-crops and normalizes the incoming BGR frame to fit the 
        exact input tensor dimensions required by the ONNX model.
        """
        h, w = frame_bgr.shape[:2]
        min_dim = min(w, h)
        self.crop_x = (w - min_dim) // 2
        self.crop_y = (h - min_dim) // 2
        
        frame_cropped = frame_bgr[self.crop_y:self.crop_y + min_dim, self.crop_x:self.crop_x + min_dim]
        self.current_crop_size = min_dim

        img_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h))
        
        data = img_resized.astype(np.float32) / 255.0
        return np.expand_dims(data, axis=0)

    def run_inference(self, input_data):
        outputs = self.session.run(None, {self.input_name: input_data})
        return outputs[0][0]

    def draw_overlay(self, frame, detected, best_pt, best_score, best_box=None):
        """
        Renders a minimalist telemetry UI on the video frame, utilizing 
        anti-aliasing and subtle color palettes to maintain visual clarity.
        """
        with self.traj_lock:
            pts = list(self.trajectory)

        # Draw trajectory tail
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = int(1 + 0.2 * alpha)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

        # Draw target bounding box
        if detected and best_pt is not None and best_box is not None:
            cx, cy = best_pt
            xmin, ymin, xmax, ymax = best_box
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        else:
            cv2.putText(frame, "STATUS: SEARCHING...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        target_x = TARGET_X_CAM0 if self.index == 0 else TARGET_X_CAM1
        cv2.line(frame, (target_x, 0), (target_x, self.frame_h), (0, 255, 255), 1, cv2.LINE_AA)
        
        # Deadzone logic mapping
        if self.index == 0:
            lim_left = target_x - YAW_DEADZONE_R
            lim_right = target_x + YAW_DEADZONE_R
        else:
            lim_left = target_x - YAW_DEADZONE_L
            lim_right = target_x + YAW_DEADZONE_R
            
        color_subtle = (90, 90, 90)
        cv2.line(frame, (lim_left, 0), (lim_left, self.frame_h), color_subtle, 1, cv2.LINE_AA)
        cv2.line(frame, (lim_right, 0), (lim_right, self.frame_h), color_subtle, 1, cv2.LINE_AA)
        
        # Descent threshold (Camera 1 exclusively)
        if self.index == 1:
            cv2.line(frame, (0, CY_TRIGGER_DESCENT), (self.frame_w, CY_TRIGGER_DESCENT), color_subtle, 1, cv2.LINE_AA)
            
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        cv2.rectangle(frame, (10, 10), (190, 55), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (190, 55), (100, 100, 100), 1)
        
        cv2.putText(frame, f"REC CAM{self.index} | {elapsed_ms} ms", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Infer: {self.last_infer_ms:.1f}ms | Track pts: {self.total_tracked_points}", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    def _write_paced(self, frame):
        now = time.time()
        if self._record_start_time is None:
            self._record_start_time = now

        due_time = self._record_start_time + self._next_frame_slot / self.fps
        while now >= due_time:
            self.writer.write(frame)
            self._next_frame_slot += 1
            due_time = self._record_start_time + self._next_frame_slot / self.fps

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print(f"[WARNING] Camera {self.index} frame drop.")
                self.stop_event.set()
                break

            t0 = time.time()
            input_data = self.preprocess(frame)              
            data_out = self.run_inference(input_data)        
            self.last_infer_ms = (time.time() - t0) * 1000   

            detected = False
            best_score = 0.0
            best_pt = None
            best_box = None  
            cx, cy, bbox_width, bbox_height = "", "", "", ""
            
            if data_out is not None and len(data_out) > 0:
                scores = data_out[:, 4]
                max_idx = np.argmax(scores)
                best_score = float(scores[max_idx])

                if best_score >= CONFIDENCE_THRESHOLD:
                    detected = True
                    
                    x_min_norm = data_out[max_idx, 0]
                    y_min_norm = data_out[max_idx, 1]
                    x_max_norm = data_out[max_idx, 2]
                    y_max_norm = data_out[max_idx, 3]
                    
                    x_min = int(x_min_norm * self.current_crop_size) + self.crop_x
                    y_min = int(y_min_norm * self.current_crop_size) + self.crop_y
                    x_max = int(x_max_norm * self.current_crop_size) + self.crop_x
                    y_max = int(y_max_norm * self.current_crop_size) + self.crop_y
                    
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    
                    best_pt = (cx, cy)
                    best_box = (x_min, y_min, x_max, y_max)
                    
                    with self.traj_lock:
                        self.trajectory.append(best_pt)
                        self.total_tracked_points += 1
                        if len(self.trajectory) > MAX_TRAJECTORY_LEN:
                            self.trajectory.pop(0)
                    with self.frame_lock:
                        self.latest_height = bbox_height if bbox_height != "" else 0
                        self.latest_cx = cx if cx != "" else 0
                        self.latest_cy = cy if cy != "" else 0
                            
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                ratio_str = 0
                if bbox_height != "" and bbox_height > 0 and cy != "":
                    ratio_val = (CAM_RES_Y - cy) / bbox_height
                    ratio_str = f"{ratio_val:.3f}".replace('.', ',')
                self.csv_writer.writerow([elapsed_ms, cx, cy, bbox_width, bbox_height, ratio_str])

            if DEBUG_PRINT_EVERY_N_FRAMES and self.frame_count % DEBUG_PRINT_EVERY_N_FRAMES == 0:
                print(f"[DEBUG] Cam {self.index} frame {self.frame_count}: "
                      f"score={best_score:.3f} | detected={detected} | infer={self.last_infer_ms:.1f}ms")

            self.draw_overlay(frame, detected, best_pt, best_score, best_box)
            self._write_paced(frame)
            self.frame_count += 1

            with self.frame_lock:
                self.latest_frame = frame

    def get_latest_frame(self):
        with self.frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def release(self):
        self.cap.release()
        self.writer.release()
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()


def main():
    stop_event = threading.Event()
    workers = []
    
    try:
        for idx in CAMERA_INDICES:
            workers.append(CameraWorker(idx, stop_event))
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        for w in workers:
            w.release()
        return

    for w in workers:
        w.start()
        
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"[INFO] Serial interface connected on {SERIAL_PORT}")
    except Exception as e:
        print(f"[ERROR] Failed to open serial port: {e}")
        ser = None

    STATE_TAKEOFF = 0
    STATE_HOVER = 1
    STATE_DESCENT = 2
    STATE_FLARE = 3
    STATE_LANDED = 4
    
    current_state = STATE_TAKEOFF
    
    hover_frames = 0
    descent_frames = 0
    flare_frames = 0
    landed_frames = 0
    
    throttle_cmd = THROTTLE_START 
    pitch_cmd = PITCH_NEUTRAL
    yaw_cmd = YAW_NEUTRAL
    
    last_serial_time = time.time()
    flight_start_time = time.time()
    hover_start_time = 0      
    descent_start_time = 0  
    flare_start_time = 0
    cam1_transition_time = 0

    active_cam = 0               
    frames_cam1_detect = 0       

    yaw_state = "IDLE"          
    end_turn_time = 0
    end_cooldown_time = 0
    current_yaw_pulse = YAW_NEUTRAL
    yaw_memory_offset = 0.0    
    
    control_log_file = open("auto-commands.csv", mode='w', newline='')
    control_csv_writer = csv.writer(control_log_file, delimiter=';')
    control_csv_writer.writerow(["time_ms", "throttle", "yaw", "pitch", "trim", "state", "active_cam"])
    start_time_control = time.time()

    print("[INFO] Flight protocol initiated. Press 'q' or ESC for emergency kill-switch.")

    try:
        while not stop_event.is_set():
            # 1. DISPLAY WINDOWS
            for w in workers:
                frame = w.get_latest_frame()
                if frame is not None:
                    cv2.imshow(w.window_name, frame)

            # 2. READ TELEMETRY FROM BOTH CAMERAS
            cam0_height, cam0_cx, cam0_cy = 0, 0, 0
            cam1_height, cam1_cx, cam1_cy = 0, 0, 0
            
            for w in workers:
                if w.index == 0: 
                    with w.frame_lock:
                        cam0_height = w.latest_height
                        cam0_cx = w.latest_cx
                        cam0_cy = w.latest_cy
                elif w.index == 1:
                    with w.frame_lock:
                        cam1_height = w.latest_height
                        cam1_cx = w.latest_cx
                        cam1_cy = w.latest_cy

            # 3. CAMERA HANDOVER LOGIC
            if active_cam == 0:
                if current_state >= STATE_HOVER and (time.time() - hover_start_time) >= MIN_TRANSITION_SEC:
                    if cam1_cx > 0:
                        frames_cam1_detect += 1
                        if frames_cam1_detect >= CAM_TRANSITION_FRAMES:
                            active_cam = 1
                            cam1_transition_time = time.time()
                            print(f"[STATE] Handover complete. Ground Camera (CAM1) has navigation control.")
                    else:
                        frames_cam1_detect = 0
                else:
                    frames_cam1_detect = 0

            # CALCULATE INDEPENDENT RATIOS
            current_ratio_cam0 = 0
            if cam0_height > 0 and cam0_cy > 0:
                current_ratio_cam0 = (CAM_RES_Y - cam0_cy) / cam0_height

            current_ratio_cam1 = 0
            if cam1_height > 0 and cam1_cy > 0:
                current_ratio_cam1 = (CAM_RES_Y - cam1_cy) / cam1_height

            # 4. YAW PULSE CONTROL WITH ODOMETRY
            active_cx = cam1_cx if active_cam == 1 else cam0_cx
            active_target_x = TARGET_X_CAM1 if active_cam == 1 else TARGET_X_CAM0

            if active_cx > 0:
                error_x = active_target_x - active_cx

                if active_cam == 0:
                    error_x = -0.5 * error_x
                
                if yaw_state == "IDLE":
                    out_of_zone = False
                    if active_cam == 1:
                        if error_x > YAW_DEADZONE_L or error_x < -YAW_DEADZONE_R:
                            out_of_zone = True
                    else:
                        if abs(error_x) > YAW_DEADZONE_R:
                            out_of_zone = True

                    if out_of_zone:
                        if (error_x > 0 and yaw_memory_offset >= YAW_MAX_MEMORY) or \
                           (error_x < 0 and yaw_memory_offset <= -YAW_MAX_MEMORY):
                            yaw_cmd = YAW_NEUTRAL
                        else:
                            turn_time = min(abs(error_x) * YAW_K_TIME, YAW_MAX_TIME)
                            
                            if error_x > 0:
                                turn_time = min(turn_time, YAW_MAX_MEMORY - yaw_memory_offset)
                                current_yaw_pulse = YAW_NEUTRAL + YAW_TURN_POWER
                                yaw_memory_offset += (turn_time * YAW_RECOVERY_FACTOR)
                            else:
                                turn_time = min(turn_time, YAW_MAX_MEMORY - abs(yaw_memory_offset))
                                current_yaw_pulse = YAW_NEUTRAL - YAW_TURN_POWER
                                yaw_memory_offset -= (turn_time * YAW_RECOVERY_FACTOR)
                                
                            end_turn_time = time.time() + turn_time
                            yaw_cmd = current_yaw_pulse
                            yaw_state = "TURNING"
                            
                    elif abs(yaw_memory_offset) > 0.05: 
                        unwind_time = min(abs(yaw_memory_offset), YAW_MAX_TIME)
                        end_turn_time = time.time() + unwind_time
                        
                        if yaw_memory_offset > 0:
                            current_yaw_pulse = YAW_NEUTRAL - YAW_TURN_POWER 
                            yaw_memory_offset -= unwind_time
                        else:
                            current_yaw_pulse = YAW_NEUTRAL + YAW_TURN_POWER 
                            yaw_memory_offset += unwind_time
                            
                        yaw_cmd = current_yaw_pulse
                        yaw_state = "TURNING"
                    else:
                        yaw_cmd = YAW_NEUTRAL
                        
                elif yaw_state == "TURNING":
                    if time.time() >= end_turn_time:
                        yaw_cmd = YAW_NEUTRAL
                        end_cooldown_time = time.time() + YAW_COOLDOWN
                        yaw_state = "COOLDOWN"
                    else:
                        yaw_cmd = current_yaw_pulse
                        
                elif yaw_state == "COOLDOWN":
                    if time.time() >= end_cooldown_time:
                        yaw_state = "IDLE"
                    yaw_cmd = YAW_NEUTRAL
            else:
                yaw_cmd = YAW_NEUTRAL
                yaw_state = "IDLE"

            # --- Extreme Proximity Kill-Switch (CAM1 Exclusively) ---
            if current_state != STATE_LANDED:
                if cam1_cy >= CY_GROUND_EMERGENCY and cam1_height > HEIGHT_GROUND_EMERGENCY:
                    landed_frames += 1
                    if landed_frames >= CONFIRMATION_FRAMES + 1:
                        print(f"[CRITICAL] Impact imminent (cy:{cam1_cy}, h:{cam1_height}px). Motors killed.")
                        current_state = STATE_LANDED
                else:
                    landed_frames = 0
            
            # 5. FINITE STATE MACHINE (Throttle Control)
            if current_state == STATE_TAKEOFF:
                elapsed_time_sec = time.time() - flight_start_time
                if elapsed_time_sec < TAKEOFF_RAMP_TIME:
                    ramp_proportion = elapsed_time_sec / TAKEOFF_RAMP_TIME
                    throttle_cmd = int(THROTTLE_START + (THROTTLE_TAKEOFF - THROTTLE_START) * ramp_proportion)
                else:
                    throttle_cmd = THROTTLE_TAKEOFF
                    print(f"[STATE] Takeoff complete ({TAKEOFF_RAMP_TIME}s). Transitioning to HOVER.")
                    current_state = STATE_HOVER
                    hover_start_time = time.time()
                    descent_frames = 0
                    
            elif current_state == STATE_HOVER:
                if active_cam == 0:
                    if cam0_height > 0 and cam0_cy > 0:
                        alt_error = TARGET_RATIO_ALT_CAM0 - current_ratio_cam0
                        throttle_cmd = int(THROTTLE_HOVER_BASE + (alt_error * KP_ALT))
                        throttle_cmd = max(THROTTLE_HOVER_BASE - 15, min(THROTTLE_HOVER_BASE + 15, throttle_cmd))
                    else:
                        throttle_cmd = THROTTLE_HOVER_BASE  
                    descent_frames = 0 
                    
                else:
                    if cam1_height > 0 and cam1_cy > 0:
                        alt_error = TARGET_RATIO_ALT_CAM1 - current_ratio_cam1
                        throttle_cmd = int(THROTTLE_HOVER_BASE + (alt_error * KP_ALT))
                        throttle_cmd = max(THROTTLE_HOVER_BASE - 15, min(THROTTLE_HOVER_BASE + 15, throttle_cmd))
                        
                        if (time.time() - cam1_transition_time) >= HOVER_STABILIZE_CAM1_SEC:
                            if cam1_cy <= CY_TRIGGER_DESCENT:
                                descent_frames += 1
                                if descent_frames >= CONFIRMATION_FRAMES:
                                    print(f"[STATE] Ceiling reached (cy:{cam1_cy}). Initiating DESCENT.")
                                    current_state = STATE_DESCENT
                                    descent_start_time = time.time()
                            else:
                                descent_frames = 0
                        else:
                            descent_frames = 0
                        
            elif current_state == STATE_DESCENT:
                elapsed_time_sec = time.time() - descent_start_time
                
                if elapsed_time_sec >= DESCENT_TIMEOUT_SEC:
                    print(f"[STATE] Descent timeout ({DESCENT_TIMEOUT_SEC}s). Motors killed for safety.")
                    current_state = STATE_LANDED
                
                elif cam1_height > 0 and cam1_cy > 0:
                    if elapsed_time_sec < DESCENT_RAMP_TIME:
                        proportion = elapsed_time_sec / DESCENT_RAMP_TIME
                        throttle_cmd = int(THROTTLE_DESCENT_START + (THROTTLE_DESCENT_END - THROTTLE_DESCENT_START) * proportion)
                    else:
                        throttle_cmd = THROTTLE_DESCENT_END
                    
                    if current_ratio_cam1 <= RATIO_TRIGGER_FLARE and elapsed_time_sec >= DESCENT_MIN_SEC:
                        flare_frames += 1
                        if flare_frames >= CONFIRMATION_FRAMES:
                            print(f"[STATE] Ground proximity detected. Executing FLARE.")
                            current_state = STATE_FLARE
                            flare_start_time = time.time()  
                    else:
                        flare_frames = 0
                        
            elif current_state == STATE_FLARE:
                throttle_cmd = THROTTLE_FLARE
                
                if (time.time() - flare_start_time) >= FLARE_DELAY_SEC:
                    print(f"[STATE] Flare duration completed ({FLARE_DELAY_SEC}s). Touchdown.")
                    current_state = STATE_LANDED
                        
            elif current_state == STATE_LANDED:
                throttle_cmd = 0
                yaw_cmd = YAW_NEUTRAL

            # 6. SERIAL TRANSMISSION
            now = time.time()
            if ser and (now - last_serial_time) >= SERIAL_INTERVAL:
                packet = bytearray([0xFF, 0xAA, int(throttle_cmd), int(yaw_cmd), int(pitch_cmd), int(TRIM_NEUTRAL)])
                ser.write(packet)
                last_serial_time = now

                elapsed_ms = int((now - start_time_control) * 1000)
                control_csv_writer.writerow([elapsed_ms, int(throttle_cmd), int(yaw_cmd), int(pitch_cmd), int(TRIM_NEUTRAL), current_state, active_cam])

            # 7. EMERGENCY KILL-SWITCH
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Emergency kill-switch activated.")
                stop_event.set()

    finally:
        stop_event.set()
        if ser:
            ser.write(bytearray([0xFF, 0xAA, 0, 63, 63, 63]))
            ser.close()

        if 'control_log_file' in locals() and not control_log_file.closed:
            control_log_file.close()
            
        for w in workers:
            w.join(timeout=2.0)
            w.release()
        cv2.destroyAllWindows()
        print("\n[OK] Flight interface safely terminated.")

if __name__ == "__main__":
    main()
