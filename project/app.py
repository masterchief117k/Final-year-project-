import os
import cv2
import time
import numpy as np
import torch
import threading
import queue
from collections import deque
from flask import Flask, render_template, request, redirect, url_for, Response, session, jsonify
from flask_socketio import SocketIO, emit
import sqlite3
from flask_bcrypt import Bcrypt
import warnings

# Suppress PyTorch deprecation warnings from YOLOv5
warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__)
app.secret_key = "ivss_super_secret"
socketio = SocketIO(app)
bcrypt = Bcrypt(app)

# --- Database Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(SCRIPT_DIR, "database.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY, name TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY, type TEXT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS employees
                 (id INTEGER PRIMARY KEY, name TEXT, emp_id TEXT UNIQUE, image_path TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        hashed_pw = bcrypt.generate_password_hash("password123").decode('utf-8')
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", hashed_pw))
        
    conn.commit()
    conn.close()

init_db()

# =============================================
#  MODEL LOADING
# =============================================

# --- 1. Face Detection Model (SSD ResNet) ---
model_path = os.path.join(SCRIPT_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
config_path = os.path.join(SCRIPT_DIR, "models", "deploy.prototxt")
if os.path.exists(model_path) and os.path.exists(config_path):
    face_net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    print("[INFO] Face detection model loaded.")
else:
    face_net = None
    print("[WARNING] Face detection models not found in models/ directory!")

# --- 2. Face Embedding Model (OpenFace) ---
embedder_path = os.path.join(SCRIPT_DIR, "models", "nn4.small2.v1.t7")
if os.path.exists(embedder_path):
    embedder_net = cv2.dnn.readNetFromTorch(embedder_path)
    print("[INFO] Face embedding model loaded.")
else:
    embedder_net = None
    print("[WARNING] Face embedding model not found! Recognition will be disabled.")

# --- 3. Weapon Detection Model (YOLOv5) ---
yolo_net = None
WEAPON_CLASSES = {"knife"}  # YOLOv5 COCO class for knife is exactly "knife" (class 43)
WEAPON_CONFIDENCE = 0.15

try:
    print("[INFO] Loading YOLOv5s model for weapon detection (this may take a moment)...")
    yolo_net = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_net.conf = WEAPON_CONFIDENCE  # NMS confidence threshold
    yolo_net.classes = [43]  # COCO class index for 'knife'
    
    print(f"[INFO] YOLOv5 loaded. Monitoring for weapon classes: {WEAPON_CLASSES}")
    print(f"[INFO] Weapon confidence threshold: {WEAPON_CONFIDENCE}")
except Exception as e:
    print(f"[WARNING] YOLOv5 failed to load! Weapon detection disabled. Error: {e}")

# --- Load Known Faces ---
known_face_names = []
known_face_embeddings = []

def load_known_faces():
    global known_face_names, known_face_embeddings
    known_face_names.clear()
    known_face_embeddings.clear()
    
    employees_dir = os.path.join(SCRIPT_DIR, "static", "employees")
    if not os.path.exists(employees_dir):
        os.makedirs(employees_dir)
        return

    # Build a map of normalized image_path -> employee name from DB
    name_map = {}
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT name, image_path FROM employees")
        for row in c.fetchall():
            name_map[os.path.normpath(row[1])] = row[0]
        conn.close()
    except Exception as e:
        print(f"[WARNING] Could not read employees from DB: {e}")

    print(f"[INFO] DB name_map has {len(name_map)} entries: {list(name_map.keys())}")

    for filename in os.listdir(employees_dir):
        filepath = os.path.normpath(os.path.join(employees_dir, filename))

        # Look up the proper name from DB; fallback to cleaned filename
        if filepath in name_map:
            name = name_map[filepath]
        else:
            # Fallback: strip emp_id prefix (format: EMPID_Name.jpg)
            raw = os.path.splitext(filename)[0]
            parts = raw.split('_', 1)
            name = parts[1].replace('_', ' ') if len(parts) > 1 else raw.replace('_', ' ')
            print(f"[WARNING] '{filepath}' not found in DB name_map, using fallback name: '{name}'")

        img = cv2.imread(filepath)
        if img is None:
            print(f"[WARNING] Could not read image: {filepath}")
            continue
        
        (h, w) = img.shape[:2]
        print(f"[INFO]   Processing '{filename}' ({w}x{h}) for face: '{name}'")
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        
        max_confidence = 0
        best_box = None
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3 and confidence > max_confidence:
                max_confidence = confidence
                best_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                
        if best_box is not None:
            (startX, startY, endX, endY) = best_box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            
            face_roi = img[startY:endY, startX:endX]
            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder_net.setInput(face_blob)
                vec = embedder_net.forward()
                
                known_face_names.append(name)
                known_face_embeddings.append(vec.flatten())
                print(f"[INFO]   ✓ Loaded face: '{name}' (detection confidence: {max_confidence:.2f})")
        else:
            print(f"[WARNING] ✗ NO FACE detected in '{filename}' — this employee will NOT be recognized!")

if face_net and embedder_net:
    print("[INFO] Loading employee faces...")
    load_known_faces()
    print(f"[INFO] Loaded {len(known_face_names)} employee(s).")

# --- Webcam Setup (replaced by threaded FrameBuffer) ---
# camera = cv2.VideoCapture(0)  # OLD: inline capture

# =============================================
#  THREADED ARCHITECTURE — LIFO FRAME BUFFER
# =============================================

class FrameBuffer:
    """Asynchronous LIFO-1 frame buffer. Daemon thread captures frames;
    consumers always get the freshest frame, preventing buffer bloat."""

    def __init__(self, src=0):
        self._cap = cv2.VideoCapture(src)
        self._lock = threading.Lock()
        self._frame = None
        self._running = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()
        return self

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame  # always overwrite — LIFO maxsize=1

    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self._cap.release()

# =============================================
#  SHARED STATE (thread-safe)
# =============================================
frame_buffer = FrameBuffer(0)

# Thread-safe result containers
identity_lock = threading.Lock()
identity_results = []          # list of dicts: {name, confidence, box, color}

weapon_lock = threading.Lock()
weapon_results = []            # list of dicts: {class, confidence, box}

# Centralized alert bus — threads push alerts, main thread emits via SocketIO
alert_bus = queue.Queue()

# Face data lock (protects known_face_names/embeddings during reload)
face_data_lock = threading.Lock()

# Cooldowns
last_face_alert_time = 0
last_weapon_alert_time = 0
last_camera_blocked_alert_time = 0
FACE_ALERT_COOLDOWN = 5
WEAPON_ALERT_COOLDOWN = 5
CAMERA_BLOCKED_COOLDOWN = 15

# Tampering
camera_dark_start = None
CAMERA_DARK_THRESHOLD = 35
CAMERA_BLOCKED_DURATION = 5

# =============================================
#  TEMPORAL CONFIDENCE ACCUMULATOR
# =============================================

class TemporalAccumulator:
    """Tracks detection confidence over k consecutive frames.
    Only triggers when avg confidence > threshold across the window."""

    def __init__(self, k=3, threshold=0.6):
        self.k = k
        self.threshold = threshold
        self._window = deque(maxlen=k)

    def update(self, detections):
        """Push frame detections. Returns filtered detections that pass temporal gate."""
        if detections:
            max_conf = max(d['confidence'] for d in detections)
            self._window.append(max_conf)
        else:
            self._window.append(0.0)

        if len(self._window) == self.k and (sum(self._window) / self.k) >= self.threshold:
            return detections  # temporally confirmed
        return []

    def reset(self):
        self._window.clear()

# =============================================
#  THREAD A — IDENTITY PIPELINE
# =============================================

class IdentityThread(threading.Thread):
    """Reads from FrameBuffer, runs SSD face detection + OpenFace embedding,
    compares against known employees via L2 Euclidean distance."""

    def __init__(self, fbuf, face_net, embedder_net):
        super().__init__(daemon=True)
        self._fbuf = fbuf
        self._face_net = face_net
        self._embedder_net = embedder_net

    def run(self):
        global last_face_alert_time
        while True:
            frame = self._fbuf.get()
            if frame is None:
                time.sleep(0.01)
                continue

            if self._face_net is None:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self._face_net.setInput(blob)
            detections = self._face_net.forward()

            results = []
            current_time = time.time()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)

                    name = "Unknown"
                    color = (0, 0, 255)

                    face_roi = frame[startY:endY, startX:endX]
                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0 and self._embedder_net:
                        face_blob = cv2.dnn.blobFromImage(
                            face_roi, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
                        self._embedder_net.setInput(face_blob)
                        vec = self._embedder_net.forward().flatten()

                        min_dist = float("inf")
                        best_match = None
                        with face_data_lock:
                            for j, known_vec in enumerate(known_face_embeddings):
                                dist = np.linalg.norm(vec - known_vec)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match = known_face_names[j]

                        if min_dist < 1.0 and best_match is not None:
                            name = best_match
                            color = (0, 255, 0)

                    results.append({
                        'name': name, 'confidence': float(confidence),
                        'box': (startX, startY, endX, endY), 'color': color
                    })

                    # Throttled alerts via bus
                    if (current_time - last_face_alert_time) > FACE_ALERT_COOLDOWN:
                        if name == "Unknown":
                            alert_bus.put({'message': 'UNKNOWN FACE detected in camera feed!',
                                           'type': 'danger', 'db_type': 'Security',
                                           'db_msg': 'Unknown face detected'})
                        else:
                            alert_bus.put({'message': f'Attendance logged for {name}',
                                           'type': 'success', 'db_type': None,
                                           'db_msg': None, 'attendance': name})
                        last_face_alert_time = current_time

            with identity_lock:
                identity_results.clear()
                identity_results.extend(results)

            time.sleep(0.01)

# =============================================
#  THREAD B — WEAPON DETECTION PIPELINE
# =============================================

class WeaponThread(threading.Thread):
    """Reads from FrameBuffer, runs YOLOv5s weapon detection with
    temporal confidence accumulation over k=3 consecutive frames."""

    def __init__(self, fbuf, yolo_net):
        super().__init__(daemon=True)
        self._fbuf = fbuf
        self._yolo = yolo_net
        self._accumulator = TemporalAccumulator(k=3, threshold=0.6)

    def run(self):
        global last_weapon_alert_time
        while True:
            frame = self._fbuf.get()
            if frame is None or self._yolo is None:
                time.sleep(0.1)
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._yolo(img_rgb)

            raw_dets = []
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det
                conf = float(conf)
                if conf >= 0.15:
                    class_name = self._yolo.names[int(cls)]
                    raw_dets.append({
                        'class': class_name.upper(),
                        'confidence': conf,
                        'box': (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    })

            confirmed = self._accumulator.update(raw_dets)

            with weapon_lock:
                weapon_results.clear()
                weapon_results.extend(confirmed)

            # Throttled alert
            current_time = time.time()
            if confirmed and (current_time - last_weapon_alert_time) > WEAPON_ALERT_COOLDOWN:
                names = list(set(d['class'] for d in confirmed))
                wstr = ", ".join(names)
                alert_bus.put({'message': f'WEAPON DETECTED: {wstr}! Immediate action required!',
                               'type': 'danger', 'db_type': 'Weapon',
                               'db_msg': f'Weapon detected: {wstr}'})
                last_weapon_alert_time = current_time

            time.sleep(0.01)

# =============================================
#  ROUTES
# =============================================

@app.route('/')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    
    if row and bcrypt.check_password_hash(row[0], password):
        session['user'] = username
        return redirect(url_for('dashboard'))
    return "Invalid Access", 401

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/api/logs')
def api_logs():
    if 'user' not in session:
        return {'logs': []}, 401
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT type, message, timestamp FROM alerts ORDER BY id DESC LIMIT 50")
    alerts = c.fetchall()
    
    c.execute("SELECT name, timestamp FROM attendance ORDER BY id DESC LIMIT 50")
    attendance = c.fetchall()
    conn.close()
    
    logs = []
    for a in alerts:
        logs.append({
            'type': a[0],
            'message': a[1],
            'timestamp': a[2]
        })
    for a in attendance:
        logs.append({
            'type': 'success',
            'message': f'Attendance logged for {a[0]}',
            'timestamp': a[1]
        })
        
    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return {'logs': logs[:50]}

# =============================================
#  HELPER: Log alert to DB
# =============================================
def log_alert_to_db(alert_type, message):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO alerts (type, message) VALUES (?, ?)", (alert_type, message))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] {e}")

# =============================================

@app.route('/api/employees', methods=['GET'])
def api_list_employees():
    if 'user' not in session:
        return jsonify({'employees': []}), 401
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, emp_id, created_at FROM employees ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify({'employees': [{'name': r[0], 'emp_id': r[1], 'created_at': r[2]} for r in rows]})

@app.route('/api/employee', methods=['POST'])
def api_add_employee():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    name = request.form.get('name', '').strip()
    emp_id = request.form.get('emp_id', '').strip()
    photo = request.files.get('photo')
    if not name or not emp_id or not photo:
        return jsonify({'error': 'Name, Employee ID, and photo are required.'}), 400

    employees_dir = os.path.join(SCRIPT_DIR, 'static', 'employees')
    os.makedirs(employees_dir, exist_ok=True)
    safe_name = name.replace(' ', '_')
    filename = f"{emp_id}_{safe_name}.jpg"
    filepath = os.path.normpath(os.path.join(employees_dir, filename))

    # Read uploaded file through OpenCV and re-encode as proper JPEG
    import tempfile
    tmp_path = os.path.join(employees_dir, f"_tmp_{filename}")
    photo.save(tmp_path)
    img = cv2.imread(tmp_path)
    if img is None:
        os.remove(tmp_path)
        return jsonify({'error': 'Could not read the uploaded image. Please upload a JPG or PNG photo.'}), 400
    cv2.imwrite(filepath, img)
    if os.path.exists(tmp_path) and os.path.normpath(tmp_path) != os.path.normpath(filepath):
        os.remove(tmp_path)

    # Validate that a face is detectable in the uploaded photo
    if face_net:
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        face_found = False
        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > 0.3:
                face_found = True
                break
        if not face_found:
            os.remove(filepath)
            return jsonify({'error': 'No face detected in the uploaded photo. Please upload a clear, front-facing photo with good lighting.'}), 400
        print(f"[INFO] Face validated in uploaded photo for '{name}'.")

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO employees (name, emp_id, image_path) VALUES (?, ?, ?)",
                  (name, emp_id, filepath))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        os.remove(filepath)
        return jsonify({'error': f'Employee ID "{emp_id}" already exists.'}), 409

    if face_net and embedder_net:
        load_known_faces()
        print(f"[INFO] Employee '{name}' added. {len(known_face_names)} face(s) now loaded.")

    return jsonify({'success': True, 'name': name, 'emp_id': emp_id})

@app.route('/api/employee', methods=['DELETE'])
def api_remove_employee():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    emp_id = (data.get('emp_id') or '').strip()
    if not emp_id:
        return jsonify({'error': 'Employee ID is required.'}), 400

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, image_path FROM employees WHERE emp_id=?", (emp_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Employee not found.'}), 404

    emp_name, image_path = row[0], row[1]
    c.execute("DELETE FROM employees WHERE emp_id=?", (emp_id,))
    conn.commit()
    conn.close()

    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    if face_net and embedder_net:
        load_known_faces()
        print(f"[INFO] Employee '{emp_name}' removed. Faces reloaded.")

    return jsonify({'success': True, 'name': emp_name})

# =============================================
#  MAIN THREAD — GENERATE FRAMES (Tampering + Overlay + SocketIO)
# =============================================
def generate_frames():
    global last_camera_blocked_alert_time, camera_dark_start

    while True:
        frame = frame_buffer.get()
        if frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        current_time = time.time()

        # --- Drain alert bus and emit via SocketIO ---
        while not alert_bus.empty():
            try:
                alert = alert_bus.get_nowait()
                socketio.emit('alert', {
                    'message': alert['message'],
                    'type': alert['type']
                }, namespace='/')
                # Log to DB
                if alert.get('db_type'):
                    log_alert_to_db(alert['db_type'], alert['db_msg'])
                if alert.get('attendance'):
                    try:
                        conn = sqlite3.connect(DB_NAME)
                        c = conn.cursor()
                        c.execute("INSERT INTO attendance (name) VALUES (?)",
                                  (alert['attendance'],))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        print(f"[DB ERROR] {e}")
            except queue.Empty:
                break

        # --- FEATURE 1: Camera Tampering Detection (main thread) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < CAMERA_DARK_THRESHOLD:
            if camera_dark_start is None:
                camera_dark_start = current_time
            dark_duration = current_time - camera_dark_start
            if dark_duration >= CAMERA_BLOCKED_DURATION:
                cv2.putText(frame, "!! CAMERA BLOCKED !!", (w // 2 - 200, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, f"Blocked for {dark_duration:.0f}s",
                            (w // 2 - 120, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if (current_time - last_camera_blocked_alert_time) > CAMERA_BLOCKED_COOLDOWN:
                    socketio.emit('alert', {
                        'message': f'CAMERA BLOCKED! Dark for {dark_duration:.0f}s. Possible tampering!',
                        'type': 'warning'
                    }, namespace='/')
                    log_alert_to_db("Tampering", f"Camera blocked for {dark_duration:.0f}s")
                    last_camera_blocked_alert_time = current_time
        else:
            camera_dark_start = None

        # --- OVERLAY: Draw identity results from Thread A ---
        with identity_lock:
            faces_snapshot = list(identity_results)
        for face in faces_snapshot:
            sx, sy, ex, ey = face['box']
            color = face['color']
            cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
            label = f"{face['name']} ({face['confidence']*100:.0f}%)"
            cv2.putText(frame, label, (sx, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- OVERLAY: Draw weapon results from Thread B ---
        with weapon_lock:
            weapons_snapshot = list(weapon_results)
        if weapons_snapshot:
            border_alpha = int(abs(np.sin(time.time() * 5)) * 255)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, border_alpha), 4)
            for det in weapons_snapshot:
                x, y, bw, bh = det['box']
                conf = det['confidence']
                cls = det['class']
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                weapon_label = f"WEAPON: {cls} ({conf*100:.0f}%)"
                (lw, lh), _ = cv2.getTextSize(weapon_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - lh - 10), (x + lw + 5, y), (0, 0, 200), -1)
                cv2.putText(frame, weapon_label, (x + 2, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode and yield MJPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("=" * 50)
    print("  I.V.S.S. - Intelligent Video Surveillance")
    print("  [ Advanced Multi-Threaded Architecture ]")
    print("=" * 50)
    print(f"  Face Detection  : {'ENABLED' if face_net else 'DISABLED'}")
    print(f"  Face Recognition: {'ENABLED' if embedder_net else 'DISABLED'}")
    print(f"  Weapon Detection: {'ENABLED' if yolo_net else 'DISABLED'}")
    print(f"  Camera Tamper   : ENABLED")
    print(f"  Threading       : LIFO Buffer + Identity + Weapon")
    print("=" * 50)

    # Start LIFO frame buffer (daemon capture thread)
    print("[INIT] Starting frame capture daemon...")
    frame_buffer.start()

    # Start Thread A — Identity Pipeline
    print("[INIT] Starting Identity thread (face detection + recognition)...")
    id_thread = IdentityThread(frame_buffer, face_net, embedder_net)
    id_thread.start()

    # Start Thread B — Weapon Detection Pipeline
    print("[INIT] Starting Weapon thread (YOLOv5s + temporal accumulation)...")
    wp_thread = WeaponThread(frame_buffer, yolo_net)
    wp_thread.start()

    print("[INIT] All threads started. Launching Flask/SocketIO...")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)
