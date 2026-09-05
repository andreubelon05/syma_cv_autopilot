import cv2
import numpy as np
import sys
import time
import threading
import onnxruntime as ort
import csv
import serial
# ==========================================
# CONFIGURACIÓ
# ==========================================

SERIAL_PORT = "COM5"      
BAUD_RATE = 9600
SERIAL_INTERVAL = 0.040   # 40 ms entre enviaments (25 Hz)

# --- NOU: Paràmetres globals i de seguretat ---
FRAMES_CONFIRMACIO = 2     # Confirmació global per a totes les etapes
CAM_RES_Y = 480            # Resolució vertical de la teva càmera

# --- NOU: Límits i Transicions basades en Ràtio i CY ---
RATIO_TRIGGER_HOVER = 4.1  # Enlairament -> Hover, full 4.1
CY_TRIGGER_HOVER = 200   # Seguretat Enlairament (cy <= 200px)

TARGET_RATIO_ALT = 4.7    # full 4.5
CY_TRIGGER_DESCENT = 135   # Hover -> Descens (Molt amunt, cy <= 125)

RATIO_TRIGGER_FLARE = 2.9  # Descens -> Float (Ràtio baixa fins a 2.8)
FLARE_DELAY_SEC = 0.550    # Retard de 600 ms abans de permetre tallar motors

# --- NOU: Seguretat d'Impacte Imminent ---
CY_TERRA_EMERGENCIA = 355
ALCADA_TERRA_EMERGENCIA = 110

RATIO_TRIGGER_TERRA = 1.1  # Float -> Aturat (La ràtio torna a pujar)
CY_TERRA_SEGURETAT = 325   # Seguretat Float (Molt a baix, cy >= 325)

# --- Potències de Throttle ---
THROTTLE_START = 35
THROTTLE_HOVER_BASE = 68     #68 full
THROTTLE_TAKEOFF = THROTTLE_HOVER_BASE + 3     
TAKEOFF_RAMP_TIME = 1.7    #2.2 segons default    
KP_ALT = 3

THROTTLE_DESCENT_START = THROTTLE_HOVER_BASE - 7
THROTTLE_DESCENT_END = THROTTLE_HOVER_BASE - 11   # Gas final si triga molt a arribar a terra
DESCENT_RAMP_TIME = 0.7      # Temps en segons per fer la transició de 65 a 50
DESCENT_TIMEOUT_SEC = 1.5     # <-- NOU: Temps màxim de descens absolut
DESCENT_MIN_SEC = 0.3       # <-- NOU: Temps mínim de caiguda obligatòria

THROTTLE_FLARE = THROTTLE_HOVER_BASE + 2

# --- Límits i Transicions ---
HOVER_STABILIZE_SEC = 2  # <-- NOU: Segons d'espera abans de permetre el descens

# --- NOU: Control de Yaw per Polsos (Impulsos temporals) ---
TARGET_X_CAM2 = 320       
YAW_DEADZONE = 50         # Píxels d'error on no fem res (centre acceptable)
YAW_TURN_POWER = 20       # Potència de l'impuls de gir (63 + 25 = 88)
YAW_K_TIME = 0.004        # Segons de gir per cada píxel d'error (Ex: 100px * 0.005 = 0.5 segons)
YAW_MAX_TIME = 0.2        # Temps màxim d'un gir per evitar virolles llargues
YAW_COOLDOWN = 0.3       # Segons d'espera (yaw=63) després de girar per observar el resultat
YAW_MAX_MEMORY = 0.85
# <-- NOU: Límit de segons acumulats (aprox 2 impulsos grans)
YAW_RECOVERY_FACTOR = 0.85  # <-- NOU: Només desfem el 90% del gir acumulat

PITCH_NEUTRAL = 64
YAW_NEUTRAL = 64
TRIM_NEUTRAL = 63

MODEL_PATH = "model_syma.onnx"          # <-- nom del teu model TFLite
CAMERA_INDICES = [0, 1]              # Índexs de les dues càmeres (usa list_cameras.py per identificar-los;
                                      # compte si tens OBS Studio instal·lat, la seva "OBS Virtual Camera"
                                      # també s'enumera com si fos una càmera més)
CONFIDENCE_THRESHOLD = 0.40          # Mateix llindar que a l'entrenament/validació
OUTPUT_VIDEO_TEMPLATE = "cam_0{idx}.mp4"
MAX_TRAJECTORY_LEN = 500             
DEBUG_PRINT_EVERY_N_FRAMES = 0      # cada quants frames s'imprimeix la millor puntuació per consola

# A Windows, el backend per defecte (MSMF) sovint falla la primera lectura d'una
# càmera real. DSHOW sol ser molt més fiable.
PREFERRED_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
CAMERA_WARMUP_ATTEMPTS = 8
CAMERA_WARMUP_DELAY = 0.15

# Controls manuals de càmera per índex, aplicats automàticament en obrir-la
# (així no cal reajustar-ho a mà cada vegada). Deixa buit {} o treu l'entrada
# per no tocar res en aquella càmera (p.ex. la integrada, índex 0, que no
# permet editar aquests paràmetres).
CAMERA_MANUAL_SETTINGS = {
    1: {  # webcam externa
        "autofocus": False,
        "focus": 0,
        "auto_exposure": False,   # False = mode manual
        "exposure": -8,
        "zoom": 0,
        "pan": 0,
        "tilt": 0,
    },
}
# ==========================================


def apply_manual_settings(cap, settings):
    """
    Aplica els controls manuals indicats (si el driver/backend els suporta) i
    retorna els valors que la càmera diu tenir DESPRÉS d'intentar-los fixar,
    perquè puguis comprovar si realment s'han aplicat (alguns UVC els
    ignoren o els retallen silenciosament).
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
        # Conveni típic del backend DSHOW a Windows: 0.75 = auto, 0.25 = manual.
        # (A V4L2/Linux sol ser 3 = auto, 1 = manual -- si mai portes el codi
        # a Linux, hauràs d'ajustar aquest valor.)
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
    Un fil independent per càmera: captura, preprocessa, infereix i grava
    SENSE esperar l'altra càmera. Cada worker té la seva pròpia instància
    de l'intèrpret TFLite, de manera que la inferència de les dues càmeres
    pot executar-se realment en paral·lel (la lectura de càmera i la
    inferència TFLite alliberen el GIL de Python mentre treballen).
    """

    def __init__(self, index, stop_event):
        super().__init__(daemon=True)
        self.index = index
        self.stop_event = stop_event

        # --- Model ONNX (una instància pròpia per worker) ---
        options = ort.SessionOptions()
        options.intra_op_num_threads = 6 
        
        # Assegura't que l'arxiu ONNX es diu així i està a la mateixa carpeta
        self.session = ort.InferenceSession(MODEL_PATH, options)
        self.input_name = self.session.get_inputs()[0].name
        
        # Extraiem les dimensions d'entrada
        shape = self.session.get_inputs()[0].shape
        self.input_h, self.input_w = shape[1], shape[2] 
        self.is_floating_model = True 
        self.input_scale, self.input_zero_point = 0.0, 0

            
        # --- Càmera (backend preferit + escalfament abans de donar-la per bona) ---
        self.cap = cv2.VideoCapture(index, PREFERRED_BACKEND)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(index)  # fallback al backend per defecte

        if not self.cap.isOpened():
            raise RuntimeError(f"No s'ha pogut obrir la càmera {index}")

        warm_ok = False
        for _ in range(CAMERA_WARMUP_ATTEMPTS):
            ret, _ = self.cap.read()
            if ret:
                warm_ok = True
                break
            time.sleep(CAMERA_WARMUP_DELAY)
        if not warm_ok:
            self.cap.release()
            raise RuntimeError(
                f"La càmera {index} s'obre però no dona imatge (ocupada per un altre "
                f"programa, o és un dispositiu virtual sense senyal, p.ex. OBS Virtual Camera)"
            )

        # --- Controls manuals de càmera (zoom, focus, exposició, pan, tilt...) ---
        settings = CAMERA_MANUAL_SETTINGS.get(index)
        if settings:
            applied = apply_manual_settings(self.cap, settings)
            #print(f"[INFO] Camera {index}: controls manuals demanats -> {settings}")
            #print(f"[INFO] Camera {index}: valors llegits després d'aplicar-los -> {applied}")
            #print(f"[INFO]   (si algun valor no coincideix amb el que has fixat, el driver "
                  #f"probablement l'ha ignorat o ajustat -- compara-ho abans de gravar)")

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and not np.isnan(fps) and fps > 0 else 30.0

        # --- Gravació ---
        self.out_path = OUTPUT_VIDEO_TEMPLATE.format(idx=index)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (self.frame_w, self.frame_h))
        # --- NOU: Datalogger CSV de Visió ---
        self.csv_filename = f"boxes_0{index}.csv"
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter=';')
        self.csv_writer.writerow(["temps_ms", "cx", "cy", "amplada", "alcada","ratio"])
        self.start_time = time.time()
        
        # --- Estat compartit amb el fil principal (protegit amb locks) ---
        self.window_name = f"Test de Vol - Camera {index}"
        self.trajectory = []
        self.traj_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_alcada = 0  # <-- NOU: Exposa l'alçada per al controlador PID
        self.latest_cx = 0
        self.latest_cy = 0
        
        self.frame_count = 0
        self.last_infer_ms = 0.0

        print(f"[INFO] Camera {index}: {self.frame_w}x{self.frame_h} @ {self.fps:.1f} FPS | "
              f"Entrada model: {self.input_w}x{self.input_h} "
              f"({'float32' if self.is_floating_model else 'int8'}) -> {self.out_path}")
        if not self.is_floating_model:
            print(f"[INFO] Camera {index}: quantització -> scale={self.input_scale}, "
                  f"zero_point={self.input_zero_point}")

        # Ritme d'escriptura del vídeo lligat al RELLOTGE REAL, no a "un frame per
        # volta de bucle". Si el processament va més lent que self.fps, es repeteix
        # l'últim frame les vegades que calguin per no perdre sincronia amb el temps
        # real (evita el "vídeo accelerat" quan la inferència no arriba al FPS nominal).
        self._record_start_time = None
        self._next_frame_slot = 0

    # ---- Preprocessament i inferència (idèntics a la lògica original) ----
    def preprocess(self, frame_bgr):
        # 1. Assegurem un retall quadrat perfecte des del centre basat en les dimensions reals del frame
        h, w = frame_bgr.shape[:2]
        min_dim = min(w, h)
        self.crop_x = (w - min_dim) // 2
        self.crop_y = (h - min_dim) // 2
        
        frame_cropped = frame_bgr[self.crop_y:self.crop_y + min_dim, self.crop_x:self.crop_x + min_dim]
        self.current_crop_size = min_dim

        # 2. Rescalar a la mida exacta del model (352x352)
        img_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h))
        
        if self.is_floating_model:
            data = img_resized.astype(np.float32) / 255.0
        else:
            norm = img_resized.astype(np.float32) / 255.0
            if self.input_scale:
                data = (norm / self.input_scale + self.input_zero_point).round().astype(np.int8)
            else:
                data = norm.astype(np.int8)
        return np.expand_dims(data, axis=0)

    def run_inference(self, input_data):
        # ONNX ja rep la matriu normalitzada en float32 des de la funció preprocess
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Retornem directament la matriu 2D de [2541, 5]
        return outputs[0][0]

    def clear_trajectory(self):
        with self.traj_lock:
            self.trajectory.clear()

    def draw_overlay(self, frame, detected, best_pt, best_score, best_box=None):
        with self.traj_lock:
            pts = list(self.trajectory)

        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = int(1 + 2 * alpha)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

        # Si hem detectat l'helicòpter i tenim les dimensions de la capsa
        if detected and best_pt is not None and best_box is not None:
            cx, cy = best_pt
            xmin, ymin, xmax, ymax = best_box
            
            # Dibuixem la capsa EXACTA que prediu el model
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            
            tag = f"Syma: {best_score*100:.1f}% | X:{cx} Y:{cy}"
            cv2.rectangle(frame, (xmax + 4, ymin), (xmax + 4 + len(tag) * 9, ymin + 22), (0, 0, 0), -1)
            cv2.putText(frame, tag, (xmax + 6, ymin + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "STATUS: SEARCHING...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- NOU: Dibuixar línia d'objectiu X per a la càmera de terra ---
        if self.index == 1:
            cv2.line(frame, (TARGET_X_CAM2, 0), (TARGET_X_CAM2, self.frame_h), (0, 255, 255), 1)
            cv2.putText(frame, f"TARGET X: {TARGET_X_CAM2}", (TARGET_X_CAM2 + 5, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            # Línies de la Zona Morta (Lila/Magenta)
            lim_esq = TARGET_X_CAM2 - YAW_DEADZONE
            lim_dreta = TARGET_X_CAM2 + YAW_DEADZONE
            
            cv2.line(frame, (lim_esq, 0), (lim_esq, self.frame_h), (255, 0, 255), 1)
            cv2.line(frame, (lim_dreta, 0), (lim_dreta, self.frame_h), (255, 0, 255), 1)
            cv2.putText(frame, "DEADZONE", (lim_esq + 5, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            
            cv2.line(frame, (0, CY_TRIGGER_DESCENT), (self.frame_w, CY_TRIGGER_DESCENT), (0, 165, 255), 1)
            cv2.putText(frame, f"DESCENS ({CY_TRIGGER_DESCENT})", (5, CY_TRIGGER_DESCENT - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
        # --- NOU: Càlcul del temps transcorregut en ms ---
        elapsed_ms = int((time.time() - self.start_time) * 1000)

        cv2.rectangle(frame, (10, 10), (190, 55), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (190, 55), (100, 100, 100), 1)
        
        # Mostrem el temps en lloc de l'etiqueta de test
        cv2.putText(frame, f"REC CAM{self.index} | {elapsed_ms} ms", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
        cv2.putText(frame, f"Infer: {self.last_infer_ms:.1f}ms | Track pts: {len(pts)}", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    def _write_paced(self, frame):
        """Escriu el frame al vídeo el nombre de vegades que calgui perquè la
        durada del fitxer coincideixi amb el temps real transcorregut, encara
        que el bucle de processament vagi més lent que self.fps."""
        now = time.time()
        if self._record_start_time is None:
            self._record_start_time = now

        due_time = self._record_start_time + self._next_frame_slot / self.fps
        while now >= due_time:
            self.writer.write(frame)
            self._next_frame_slot += 1
            due_time = self._record_start_time + self._next_frame_slot / self.fps

    # ---- Bucle propi del fil: captura -> infereix -> grava, sense parar-se per l'altra càmera ----
    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print(f"[AVÍS] No es reben fotogrames de la càmera {self.index}.")
                self.stop_event.set()
                break

            t0 = time.time()
            input_data = self.preprocess(frame)              # <-- AIXÒ ES QUEDA (prepara la imatge)
            data_out = self.run_inference(input_data)        # <-- AIXÒ ES QUEDA (executa la IA)
            self.last_infer_ms = (time.time() - t0) * 1000   # <-- AIXÒ ES QUEDA (calcula els ms)

            detected = False
            best_score = 0.0
            best_pt = None
            best_box = None  # Afegim la variable per guardar la capsa
            # 1. Preparem variables buides per si aquest frame no detecta res
            cx, cy, amplada, alcada = "", "", "", ""
            
            if data_out is not None and len(data_out) > 0:
                scores = data_out[:, 4]
                max_idx = np.argmax(scores)
                best_score = float(scores[max_idx])

                if best_score >= CONFIDENCE_THRESHOLD:
                    detected = True
                    
                    # El format d'Edge Impulse confirmat és: [xmin, ymin, xmax, ymax]
                    x_min_norm = data_out[max_idx, 0]
                    y_min_norm = data_out[max_idx, 1]
                    x_max_norm = data_out[max_idx, 2]
                    y_max_norm = data_out[max_idx, 3]
                    
                    # 1. Calculem directament els 4 píxels reals a la imatge de la càmera
                    x_min = int(x_min_norm * self.current_crop_size) + self.crop_x
                    y_min = int(y_min_norm * self.current_crop_size) + self.crop_y
                    x_max = int(x_max_norm * self.current_crop_size) + self.crop_x
                    y_max = int(y_max_norm * self.current_crop_size) + self.crop_y
                    
                    # 2. El centre exacte és la meitat entre els dos extrems
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    
                    amplada = x_max - x_min
                    alcada = y_max - y_min
                    
                    best_pt = (cx, cy)
                    best_box = (x_min, y_min, x_max, y_max)
                    
                    with self.traj_lock:
                        self.trajectory.append(best_pt)
                        if len(self.trajectory) > MAX_TRAJECTORY_LEN:
                            self.trajectory.pop(0)
                    with self.frame_lock:
                        # Si es perd la detecció (""), guardem un 0 per no trencar les matemàtiques
                        self.latest_alcada = alcada if alcada != "" else 0
                        self.latest_cx = cx if cx != "" else 0
                        self.latest_cy = cy if cy != "" else 0
                            
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                ratio_str = 0
                if alcada != "" and alcada > 0 and cy != "":
                    ratio_val = (CAM_RES_Y - cy) / alcada
                    # Formategem a 3 decimals i canviem el punt per la coma
                    ratio_str = f"{ratio_val:.3f}".replace('.', ',')
                self.csv_writer.writerow([elapsed_ms, cx, cy, amplada, alcada, ratio_str])

            if DEBUG_PRINT_EVERY_N_FRAMES and self.frame_count % DEBUG_PRINT_EVERY_N_FRAMES == 0:
                print(f"[DEBUG] Cam {self.index} frame {self.frame_count}: "
                      f"best_score={best_score:.3f} | detected={detected} | "
                      f"infer={self.last_infer_ms:.1f}ms")

            self.draw_overlay(frame, detected, best_pt, best_score,best_box)
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
        
    # --- OBRIR PORT SÈRIE ---
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"[INFO] Connectat a la Pico 2W pel port {SERIAL_PORT}")
    except Exception as e:
        print(f"[ERROR SÈRIE] No s'ha pogut obrir el port: {e}")
        ser = None

    # --- VARIABLES DE LA MÀQUINA D'ESTATS (Standard UAV Naming) ---
    STATE_TAKEOFF = 0
    STATE_HOVER = 1
    STATE_DESCENT = 2
    STATE_FLARE = 3
    STATE_LANDED = 4
    
    current_state = STATE_TAKEOFF
    
    # Comptadors amb el nou llindar global de 2 frames
    frames_hover = 0
    frames_descent = 0
    frames_flare = 0
    frames_landed = 0
    
    throttle_cmd = THROTTLE_START 
    pitch_cmd = PITCH_NEUTRAL
    yaw_cmd = YAW_NEUTRAL
    
    last_serial_time = time.time()
    temps_inici_vol = time.time()
    temps_inici_hover = 0      # <-- NOU: Cronòmetre d'inèrcia d'enlairament
    temps_inici_descent = 0  # <-- NOU: Cronòmetre de la rampa de caiguda
    temps_inici_flare = 0  # Temps on entra a l'etapa FLOAT

    # --- Variables del Control per Polsos (Yaw) ---
    yaw_estat = "IDLE"          # "IDLE", "TURNING", "COOLDOWN"
    temps_fi_gir = 0
    temps_fi_cooldown = 0
    yaw_pols_actual = YAW_NEUTRAL
    yaw_memoria_offset = 0.0    # <-- NOU: Acumulador de segons per desfer els girs
    
    # Inicialització del registre de control
    control_log_file = open("auto-commands.csv", mode='w', newline='')
    control_csv_writer = csv.writer(control_log_file, delimiter=';')
    control_csv_writer.writerow(["temps_ms", "throttle", "yaw", "pitch", "trim", "estat"])
    start_time_control = time.time()

    print("[INFO] Controls:")
    print("       - 'q' o ESC per aturar d'emergència i apagar motors.")

    try:
        while not stop_event.is_set():
            # 1. MOSTRAR FINESTRES
            for w in workers:
                frame = w.get_latest_frame()
                if frame is not None:
                    cv2.imshow(w.window_name, frame)

            # 2. LLEGIR TELEMETRIA DE LA CÀMERA D'ATERRATGE
            cam_terra_alcada = 0
            cam_terra_cx = 0
            cam_terra_cy = 0
            for w in workers:
                if w.index == 1: 
                    with w.frame_lock:
                        cam_terra_alcada = w.latest_alcada
                        cam_terra_cx = w.latest_cx
                        cam_terra_cy = w.latest_cy

            # CÀLCUL GLOBAL DE LA RÀTIO (Es farà servir a totes les etapes)
            relacio_actual = 0
            if cam_terra_alcada > 0 and cam_terra_cy > 0:
                y_terra = CAM_RES_Y - cam_terra_cy
                relacio_actual = y_terra / cam_terra_alcada

            # 3. CÀLCUL DEL YAW (Control per Polsos amb Memòria i Saturació)
            if cam_terra_cx > 0:
                error_x = TARGET_X_CAM2 - cam_terra_cx
                
                if yaw_estat == "IDLE":
                    # Cas A: Estem fora del centre, hem de corregir posició
                    if abs(error_x) > YAW_DEADZONE:
                        
                        # 1. Comprovem si el morro ja està girat al límit permès
                        if (error_x > 0 and yaw_memoria_offset >= YAW_MAX_MEMORY) or \
                           (error_x < 0 and yaw_memoria_offset <= -YAW_MAX_MEMORY):
                            # Saturació assolida: ens mantenim neutres esperant que la inèrcia el porti
                            yaw_cmd = YAW_NEUTRAL
                        
                        else:
                            # 2. Calculem el temps de gir i retallem si ens passem del límit
                            temps_gir = min(abs(error_x) * YAW_K_TIME, YAW_MAX_TIME)
                            
                            if error_x > 0:
                                temps_gir = min(temps_gir, YAW_MAX_MEMORY - yaw_memoria_offset)
                                yaw_pols_actual = YAW_NEUTRAL + YAW_TURN_POWER
                                yaw_memoria_offset += (temps_gir * YAW_RECOVERY_FACTOR)
                            else:
                                temps_gir = min(temps_gir, YAW_MAX_MEMORY - abs(yaw_memoria_offset))
                                yaw_pols_actual = YAW_NEUTRAL - YAW_TURN_POWER
                                yaw_memoria_offset -= (temps_gir * YAW_RECOVERY_FACTOR)
                                
                            temps_fi_gir = time.time() + temps_gir
                            yaw_cmd = yaw_pols_actual
                            yaw_estat = "TURNING"
                            print(f"[YAW] Encarant {temps_gir:.2f}s (Memòria total: {yaw_memoria_offset:.2f}s)")
                            
                    # Cas B: Estem al centre, però tenim girs acumulats per desfer
                    elif abs(yaw_memoria_offset) > 0.05: 
                        # Limitem el temps màxim de desfer per no fer virolles llargues de cop
                        temps_unwind = min(abs(yaw_memoria_offset), YAW_MAX_TIME)
                        temps_fi_gir = time.time() + temps_unwind
                        
                        if yaw_memoria_offset > 0:
                            yaw_pols_actual = YAW_NEUTRAL - YAW_TURN_POWER 
                            yaw_memoria_offset -= temps_unwind
                            direccio = "ESQ"
                        else:
                            yaw_pols_actual = YAW_NEUTRAL + YAW_TURN_POWER 
                            yaw_memoria_offset += temps_unwind
                            direccio = "DRETA"
                            
                        yaw_cmd = yaw_pols_actual
                        yaw_estat = "TURNING"
                        print(f"[YAW] Desfent {temps_unwind:.2f}s cap a {direccio}. Restant: {yaw_memoria_offset:.2f}s")
                    
                    # Cas C: Estem al centre i rectes
                    else:
                        yaw_cmd = YAW_NEUTRAL
                        
                elif yaw_estat == "TURNING":
                    if time.time() >= temps_fi_gir:
                        yaw_cmd = YAW_NEUTRAL
                        temps_fi_cooldown = time.time() + YAW_COOLDOWN
                        yaw_estat = "COOLDOWN"
                    else:
                        yaw_cmd = yaw_pols_actual
                        
                elif yaw_estat == "COOLDOWN":
                    if time.time() >= temps_fi_cooldown:
                        yaw_estat = "IDLE"
                    yaw_cmd = YAW_NEUTRAL
            else:
                yaw_cmd = YAW_NEUTRAL
                yaw_estat = "IDLE"

            # --- NOU: Kill-switch per proximitat extrema a terra ---
            if current_state != STATE_LANDED:
                if cam_terra_cy >= CY_TERRA_EMERGENCIA and cam_terra_alcada > ALCADA_TERRA_EMERGENCIA:
                    frames_landed += 1
                    if frames_landed >= FRAMES_CONFIRMACIO + 1:
                        print(f"[ESTAT] IMPACTE IMMINENT (cy:{cam_terra_cy}, capsa:{cam_terra_alcada}px). Tallant motors!")
                        current_state = STATE_LANDED
                else:
                    frames_landed = 0
            
            # 4. MÀQUINA D'ESTATS (Throttle)
            if current_state == STATE_TAKEOFF:
                temps_transcorregut = time.time() - temps_inici_vol
                if temps_transcorregut < TAKEOFF_RAMP_TIME:
                    proporcio_rampa = temps_transcorregut / TAKEOFF_RAMP_TIME
                    throttle_cmd = int(THROTTLE_START + (THROTTLE_TAKEOFF - THROTTLE_START) * proporcio_rampa)
                else:
                    throttle_cmd = THROTTLE_TAKEOFF
                    
                # Transició a HOVER (Ràtio >= 4.2 OR sostre de seguretat cy <= 200px)
                if relacio_actual >= RATIO_TRIGGER_HOVER or (0 < cam_terra_cy <= CY_TRIGGER_HOVER):
                    frames_hover += 1
                    if frames_hover >= FRAMES_CONFIRMACIO:
                        print(f"[ESTAT] Condició assolida (Ratio:{relacio_actual:.2f}). Passant a HOVER.")
                        current_state = STATE_HOVER
                        temps_inici_hover = time.time()
                        frames_descent = 0
                else:
                    frames_hover = 0
                    
            elif current_state == STATE_HOVER:
                if cam_terra_alcada > 0 and cam_terra_cy > 0:
                    
                    error_alt = TARGET_RATIO_ALT - relacio_actual
                    throttle_cmd = int(THROTTLE_HOVER_BASE + (error_alt * KP_ALT))
                    throttle_cmd = max(THROTTLE_HOVER_BASE - 15, min(THROTTLE_HOVER_BASE + 15, throttle_cmd))
                    
                    # Transició a DESCENT (Molt amunt: cy <= 130)
                    if (time.time() - temps_inici_hover) >= HOVER_STABILIZE_SEC:
                        if cam_terra_cy <= CY_TRIGGER_DESCENT:
                            frames_descent += 1
                            if frames_descent >= FRAMES_CONFIRMACIO:
                                print(f"[ESTAT] Sostre superat (cy:{cam_terra_cy}). Iniciant descens...")
                                current_state = STATE_DESCENT
                                temps_inici_descent = time.time() # <-- NOU: Disparem el cronòmetre
                        else:
                            frames_descent = 0
                    else:
                        frames_descent = 0
                        
            elif current_state == STATE_DESCENT:
                temps_transcorregut = time.time() - temps_inici_descent
                
                # 1. Condició de seguretat incondicional (sempre s'avalua, fins i tot a cegues)
                if temps_transcorregut >= DESCENT_TIMEOUT_SEC:
                    print(f"[ESTAT] Temps de descens esgotat ({DESCENT_TIMEOUT_SEC}s). Tallant motors per seguretat.")
                    current_state = STATE_LANDED
                
                elif cam_terra_alcada > 0 and cam_terra_cy > 0:
                    # Rampa de temps: baixa de 65 a 50 durant X segons
                    temps_transcorregut = time.time() - temps_inici_descent
                    if temps_transcorregut < DESCENT_RAMP_TIME:
                        proporcio = temps_transcorregut / DESCENT_RAMP_TIME
                        throttle_cmd = int(THROTTLE_DESCENT_START + (THROTTLE_DESCENT_END - THROTTLE_DESCENT_START) * proporcio)
                    else:
                        throttle_cmd = THROTTLE_DESCENT_END
                    
                    # Transició a FLARE (ratio <= 2.8) - La càmera sempre vigila
                    if relacio_actual <= RATIO_TRIGGER_FLARE and temps_transcorregut >= DESCENT_MIN_SEC:
                        frames_flare += 1
                        if frames_flare >= FRAMES_CONFIRMACIO:
                            print(f"[ESTAT] Ràtio baixa ({relacio_actual:.2f}). Esmorteint caiguda (Flare)...")
                            current_state = STATE_FLARE
                            temps_inici_flare = time.time()  
                    else:
                        frames_flare = 0
                        
            elif current_state == STATE_FLARE:
                throttle_cmd = THROTTLE_FLARE
                
                # Tallem motors incondicionalment per seguretat un cop passat el temps
                if (time.time() - temps_inici_flare) >= FLARE_DELAY_SEC:
                    print(f"[ESTAT] Temps de Flare esgotat ({FLARE_DELAY_SEC}s). Tallant motors per seguretat.")
                    current_state = STATE_LANDED
                        
            elif current_state == STATE_LANDED:
                throttle_cmd = 0
                yaw_cmd = YAW_NEUTRAL

            # 5. ENVIAMENT SÈRIE (Cada 40 ms)
            now = time.time()
            if ser and (now - last_serial_time) >= SERIAL_INTERVAL:
                paquet = bytearray([0xFF, 0xAA, int(throttle_cmd), int(yaw_cmd), int(pitch_cmd), int(TRIM_NEUTRAL)])
                ser.write(paquet)
                last_serial_time = now

                # --- NOU: Enregistrar les dades al CSV ---
                elapsed_ms = int((now - start_time_control) * 1000)
                control_csv_writer.writerow([elapsed_ms, int(throttle_cmd), int(yaw_cmd), int(pitch_cmd), int(TRIM_NEUTRAL), current_state])

            # 6. CONTROL D'ATURADA
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Aturada d'emergència iniciada.")
                stop_event.set()

    finally:
        stop_event.set()
        # Apagar motors en tancar
        if ser:
            ser.write(bytearray([0xFF, 0xAA, 0, 63, 63, 63]))
            ser.close()

        # --- NOU: Tancar l'arxiu de registre de control ---
        if 'control_log_file' in locals() and not control_log_file.closed:
            control_log_file.close()
            
        for w in workers:
            w.join(timeout=2.0)
            w.release()
        cv2.destroyAllWindows()
        print("\n[OK] Aplicació tancada de forma segura.")

if __name__ == "__main__":
    main()
