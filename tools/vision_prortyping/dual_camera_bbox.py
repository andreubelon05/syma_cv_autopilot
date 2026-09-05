import cv2
import numpy as np
import sys
import time
import threading
import onnxruntime as ort
import csv
# ==========================================
# CONFIGURACIÓ
# ==========================================
CAM_RES_Y = 480
MODEL_PATH = "model_syma.onnx"          # <-- nom del teu model TFLite
CAMERA_INDICES = [0, 1]              # Índexs de les dues càmeres (usa list_cameras.py per identificar-los;
                                      # compte si tens OBS Studio instal·lat, la seva "OBS Virtual Camera"
                                      # també s'enumera com si fos una càmera més)
CONFIDENCE_THRESHOLD = 0.40          # Mateix llindar que a l'entrenament/validació
OUTPUT_VIDEO_TEMPLATE = "cam_0{idx}.mp4"
MAX_TRAJECTORY_LEN = 100             
DEBUG_PRINT_EVERY_N_FRAMES = 15      # cada quants frames s'imprimeix la millor puntuació per consola
TARGET_X_CAM2 = 320  # Línia groga de referència per al Yaw

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
        "exposure": -9,
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
        options.intra_op_num_threads = 6 # Assignem 4 nuclis del teu Ryzen per càmera
        #options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Assegura't que l'arxiu ONNX es diu així i està a la mateixa carpeta
        self.session = ort.InferenceSession("model_B.onnx", options)
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
            print(f"[INFO] Camera {index}: controls manuals demanats -> {settings}")
            print(f"[INFO] Camera {index}: valors llegits després d'aplicar-los -> {applied}")
            print(f"[INFO]   (si algun valor no coincideix amb el que has fixat, el driver "
                  f"probablement l'ha ignorat o ajustat -- compara-ho abans de gravar)")

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and not np.isnan(fps) and fps > 0 else 30.0
        self.box_half_size = max(15, int(0.025 * self.frame_w))

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
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
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
                            
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                
                # --- NOU: Càlcul de la Ràtio ---
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

    print("[INFO] Controls:")
    print("       - 'c' per netejar les línies de trajectòria (ambdues càmeres).")
    print("       - 'q' o ESC per aturar la prova i guardar els vídeos.")

    # El fil principal NOMÉS mostra finestres i llegeix el teclat (les càmeres
    # no depenen d'ell per anar rodant al seu propi ritme).
    start_time = time.time()
    try:
        while not stop_event.is_set():
            for w in workers:
                frame = w.get_latest_frame()
                if frame is not None:
                    cv2.imshow(w.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                stop_event.set()
            elif key == ord('c'):
                for w in workers:
                    w.clear_trajectory()
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=2.0)
        for w in workers:
            w.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("\n[OK] Prova completada!")
    print(f"     - Durada total: {total_time:.1f} s")
    for w in workers:
        print(f"     - Cam {w.index}: {w.frame_count} fotogrames -> {w.out_path}")


if __name__ == "__main__":
    main()
