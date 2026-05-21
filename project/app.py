import eventlet
eventlet.monkey_patch()

import os
import cv2
import time
import base64
import numpy as np
import torch
import threading
import queue
import traceback
from io import BytesIO
from PIL import Image
from collections import deque
from flask import Flask, render_template, request, redirect, url_for, Response, session, jsonify
from flask_socketio import SocketIO, emit
import sqlite3
from flask_bcrypt import Bcrypt
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- Load environment variables ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or __import__('secrets').token_hex(32)

# Production session hardening
if os.environ.get('FLASK_ENV') == 'production':
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
    )

socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=False, engineio_logger=False)
bcrypt = Bcrypt(app)

# --- Image encryption (Fernet AES-128-CBC) ---
_raw_img_key = os.environ.get('IMAGE_ENCRYPTION_KEY')
if not _raw_img_key:
    _raw_img_key = Fernet.generate_key().decode()
    print('[SECURITY] WARNING: No IMAGE_ENCRYPTION_KEY in .env — generated ephemeral key. Images from previous runs cannot be decrypted!')
fernet = Fernet(_raw_img_key.encode() if isinstance(_raw_img_key, str) else _raw_img_key)

# --- Persistent Storage ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR: Render Disk mount (/data) in production, project dir locally
DATA_DIR = os.environ.get('DATA_DIR', SCRIPT_DIR)
os.makedirs(os.path.join(DATA_DIR, 'employees'), exist_ok=True)

# --- Database Setup ---
DB_NAME = os.path.join(DATA_DIR, "database.db")

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
        admin_pw = os.environ.get('ADMIN_PASSWORD', 'password123')
        hashed_pw = bcrypt.generate_password_hash(admin_pw).decode('utf-8')
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", hashed_pw))
        print(f'[INIT] Admin account created. Change ADMIN_PASSWORD in .env after first login.')
    conn.commit()
    conn.close()

init_db()

# =============================================
#  MODEL LOADING
# =============================================

model_path = os.path.join(SCRIPT_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
config_path = os.path.join(SCRIPT_DIR, "models", "deploy.prototxt")
if os.path.exists(model_path) and os.path.exists(config_path):
    face_net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    print("[INFO] Face detection model loaded.")
else:
    face_net = None
    print("[WARNING] Face detection models not found!")

embedder_path = os.path.join(SCRIPT_DIR, "models", "nn4.small2.v1.t7")
if os.path.exists(embedder_path):
    embedder_net = cv2.dnn.readNetFromTorch(embedder_path)
    print("[INFO] Face embedding model loaded.")
else:
    embedder_net = None
    print("[WARNING] Face embedding model not found!")

yolo_net = None
WEAPON_CLASSES = {"knife"}
WEAPON_CONFIDENCE = 0.15
try:
    print("[INFO] Loading YOLOv5s...")
    yolo_net = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_net.conf = WEAPON_CONFIDENCE
    yolo_net.classes = [43]
    print(f"[INFO] YOLOv5 loaded.")
except Exception as e:
    print(f"[WARNING] YOLOv5 failed: {e}")

# --- Known Faces ---
known_face_names = []
known_face_embeddings = []
face_data_lock = threading.Lock()

def load_known_faces():
    global known_face_names, known_face_embeddings
    known_face_names.clear()
    known_face_embeddings.clear()
    employees_dir = os.path.join(DATA_DIR, "employees")
    if not os.path.exists(employees_dir):
        os.makedirs(employees_dir)
        return
    name_map = {}
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT name, image_path FROM employees")
        for row in c.fetchall():
            name_map[os.path.normpath(row[1])] = row[0]
        conn.close()
    except Exception as e:
        print(f"[WARNING] DB read error: {e}")
    for filename in os.listdir(employees_dir):
        filepath = os.path.normpath(os.path.join(employees_dir, filename))
        if filepath in name_map:
            name = name_map[filepath]
        else:
            continue  # skip files not in DB

        # Decrypt if .enc, else try reading directly (legacy plain images)
        try:
            if filename.endswith('.enc'):
                with open(filepath, 'rb') as f:
                    encrypted_bytes = f.read()
                img_bytes = fernet.decrypt(encrypted_bytes)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(filepath)
        except Exception as e:
            print(f"[WARNING] Could not load/decrypt {filename}: {e}")
            continue

        if img is None:
            continue
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        max_confidence = 0
        best_box = None
        for i in range(detections.shape[2]):
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
                face_blob = cv2.dnn.blobFromImage(face_roi, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
                embedder_net.setInput(face_blob)
                vec = embedder_net.forward()
                known_face_names.append(name)
                known_face_embeddings.append(vec.flatten())
                print(f"[INFO] Loaded face: '{name}'")

if face_net and embedder_net:
    load_known_faces()
    print(f"[INFO] Loaded {len(known_face_names)} employee(s).")

# =============================================
#  SOCKET-DRIVEN FRAME BUFFER (per-camera)
# =============================================

class SocketFrameBuffer:
    """LIFO-1 buffer fed by SocketIO instead of cv2.VideoCapture."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def put(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

# =============================================
#  TEMPORAL CONFIDENCE ACCUMULATOR
# =============================================

class TemporalAccumulator:
    def __init__(self, k=3, threshold=0.6):
        self.k = k
        self.threshold = threshold
        self._window = deque(maxlen=k)

    def update(self, detections):
        if detections:
            max_conf = max(d['confidence'] for d in detections)
            self._window.append(max_conf)
        else:
            self._window.append(0.0)
        if len(self._window) == self.k and (sum(self._window) / self.k) >= self.threshold:
            return detections
        return []

# =============================================
#  PER-CAMERA INFERENCE MANAGER
# =============================================

class CameraSession:
    """Manages a dedicated Thread A (identity) + Thread B (weapon) pair for one camera."""

    def __init__(self, camera_id, socketio_ref):
        self.camera_id = camera_id
        self.buffer = SocketFrameBuffer()
        self.socketio = socketio_ref
        self.alert_bus = queue.Queue()
        self._running = False
        self._identity_thread = None
        self._weapon_thread = None
        # Cooldowns
        self.last_face_alert = 0
        self.last_weapon_alert = 0
        self.last_tamper_alert = 0
        self.dark_start = None
        # Daily attendance dedup: {name -> date_string}
        self._attendance_today = {}
        # FPS tracking
        self._frame_count = 0
        self._fps_start = time.time()
        self._current_fps = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        # Use socketio.start_background_task for eventlet compatibility
        socketio.start_background_task(self._identity_loop)
        socketio.start_background_task(self._weapon_loop)
        print(f"[CAMERA {self.camera_id[:12]}] Inference threads started.")

    def stop(self):
        self._running = False
        print(f"[CAMERA {self.camera_id}] Inference threads stopping.")

    def feed_frame(self, frame):
        self.buffer.put(frame)
        self._frame_count += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start = time.time()

    def _identity_loop(self):
        print(f"[THREAD-A {self.camera_id[:12]}] Identity loop started")
        while self._running:
            try:
                frame = self.buffer.get()
                if frame is None or face_net is None:
                    eventlet.sleep(0.05)
                    continue
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()
                faces = []
                now = time.time()
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (sx, sy, ex, ey) = box.astype("int")
                        sx, sy = int(max(0, sx)), int(max(0, sy))
                        ex, ey = int(min(w, ex)), int(min(h, ey))
                        name = "Unknown"
                        known = False
                        face_roi = frame[sy:ey, sx:ex]
                        if face_roi.shape[0] > 0 and face_roi.shape[1] > 0 and embedder_net:
                            fb = cv2.dnn.blobFromImage(face_roi, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
                            embedder_net.setInput(fb)
                            vec = embedder_net.forward().flatten()
                            min_dist = float("inf")
                            best = None
                            with face_data_lock:
                                for j, kv in enumerate(known_face_embeddings):
                                    dist = float(np.linalg.norm(vec - kv))
                                    if dist < min_dist:
                                        min_dist = dist
                                        best = known_face_names[j]
                            if min_dist < 1.0 and best:
                                name = best
                                known = True
                        faces.append({'name': name, 'confidence': confidence,
                                      'box': [sx, sy, ex, ey], 'known': known})
                        if (now - self.last_face_alert) > 5:
                            if not known:
                                self.alert_bus.put({'message': 'UNKNOWN FACE detected!', 'type': 'danger',
                                                    'db_type': 'Security', 'db_msg': 'Unknown face detected'})
                                self.last_face_alert = now
                            else:
                                # Daily dedup: only log attendance once per person per day
                                today = time.strftime('%Y-%m-%d')
                                last_date = self._attendance_today.get(name)
                                if last_date != today:
                                    self._attendance_today[name] = today
                                    self.alert_bus.put({'message': f'Attendance: {name}', 'type': 'success',
                                                        'db_type': None, 'attendance': name})
                                    self.last_face_alert = now

                # Tampering check
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))
                tampering = False
                if brightness < 35:
                    if self.dark_start is None:
                        self.dark_start = now
                    if (now - self.dark_start) >= 5:
                        tampering = True
                        if (now - self.last_tamper_alert) > 15:
                            self.alert_bus.put({'message': 'CAMERA BLOCKED!', 'type': 'warning',
                                                'db_type': 'Tampering', 'db_msg': 'Camera obstruction detected'})
                            self.last_tamper_alert = now
                else:
                    self.dark_start = None

                # Emit face + tampering results
                self.socketio.emit('inference_results', {
                    'camera_id': self.camera_id,
                    'faces': faces,
                    'weapons': [],
                    'tampering': tampering,
                    'fps': round(self._current_fps, 1)
                }, namespace='/')

                self._drain_alerts()
                eventlet.sleep(0.01)
            except Exception as e:
                print(f"[THREAD-A ERROR] {e}")
                traceback.print_exc()
                eventlet.sleep(0.5)

    def _weapon_loop(self):
        print(f"[THREAD-B {self.camera_id[:12]}] Weapon loop started")
        accumulator = TemporalAccumulator(k=3, threshold=0.6)
        while self._running:
            try:
                frame = self.buffer.get()
                if frame is None or yolo_net is None:
                    eventlet.sleep(0.1)
                    continue
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = yolo_net(img_rgb)
                raw_dets = []
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det
                    conf = float(conf)
                    if conf >= 0.15:
                        class_name = yolo_net.names[int(cls)]
                        raw_dets.append({'class': class_name.upper(), 'confidence': conf,
                                         'box': [int(x1), int(y1), int(x2-x1), int(y2-y1)]})
                confirmed = accumulator.update(raw_dets)
                if confirmed:
                    now = time.time()
                    self.socketio.emit('inference_results', {
                        'camera_id': self.camera_id,
                        'faces': [],
                        'weapons': [{'class': d['class'], 'confidence': d['confidence'], 'box': d['box']}
                                    for d in confirmed],
                        'tampering': False,
                        'fps': round(self._current_fps, 1)
                    }, namespace='/')
                    if (now - self.last_weapon_alert) > 5:
                        names = list(set(d['class'] for d in confirmed))
                        self.alert_bus.put({'message': f'WEAPON: {", ".join(names)}!', 'type': 'danger',
                                            'db_type': 'Weapon', 'db_msg': f'Weapon: {", ".join(names)}'})
                        self.last_weapon_alert = now
                    self._drain_alerts()
                eventlet.sleep(0.01)
            except Exception as e:
                print(f"[THREAD-B ERROR] {e}")
                traceback.print_exc()
                eventlet.sleep(0.5)

    def _drain_alerts(self):
        while not self.alert_bus.empty():
            try:
                alert = self.alert_bus.get_nowait()
                self.socketio.emit('alert', {'message': alert['message'], 'type': alert['type']}, namespace='/')
                if alert.get('db_type'):
                    log_alert_to_db(alert['db_type'], alert['db_msg'])
                if alert.get('attendance'):
                    try:
                        conn = sqlite3.connect(DB_NAME)
                        c = conn.cursor()
                        c.execute("INSERT INTO attendance (name) VALUES (?)", (alert['attendance'],))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        print(f"[DB ERROR] {e}")
            except queue.Empty:
                break

# =============================================
#  CAMERA SESSION REGISTRY
# =============================================

camera_sessions = {}  # camera_id -> CameraSession
sessions_lock = threading.Lock()

def get_or_create_session(camera_id):
    with sessions_lock:
        if camera_id not in camera_sessions:
            camera_sessions[camera_id] = CameraSession(camera_id, socketio)
            camera_sessions[camera_id].start()
        return camera_sessions[camera_id]

def remove_session(camera_id):
    with sessions_lock:
        sess = camera_sessions.pop(camera_id, None)
        if sess:
            sess.stop()

# =============================================
#  SOCKETIO EVENT HANDLERS
# =============================================

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        camera_id = data.get('camera_id', 'default')
        image_data = data.get('image', '')

        # Strip data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        # Decode base64 -> PIL -> numpy
        img_bytes = base64.b64decode(image_data)
        pil_img = Image.open(BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Feed to per-camera session
        session_obj = get_or_create_session(camera_id)
        session_obj.feed_frame(frame)
    except Exception as e:
        print(f"[FRAME ERROR] {e}")

@socketio.on('cameras_active')
def handle_cameras_active(data):
    active_ids = set(data.get('camera_ids', []))
    with sessions_lock:
        stale = [cid for cid in camera_sessions if cid not in active_ids]
    for cid in stale:
        remove_session(cid)

@socketio.on('connect')
def handle_connect():
    print(f"[WS] Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WS] Client disconnected: {request.sid}")

# =============================================
#  ROUTES
# =============================================

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'face_detection': face_net is not None,
        'face_recognition': embedder_net is not None,
        'weapon_detection': yolo_net is not None,
        'known_faces': len(known_face_names),
        'active_cameras': len(camera_sessions),
    }), 200

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
    alerts_rows = c.fetchall()
    c.execute("SELECT name, timestamp FROM attendance ORDER BY id DESC LIMIT 50")
    attendance_rows = c.fetchall()
    conn.close()
    logs = []
    for a in alerts_rows:
        logs.append({'type': a[0], 'message': a[1], 'timestamp': a[2]})
    for a in attendance_rows:
        logs.append({'type': 'success', 'message': f'Attendance logged for {a[0]}', 'timestamp': a[1]})
    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return {'logs': logs[:50]}

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
#  EMPLOYEE API (unchanged)
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
    employees_dir = os.path.join(DATA_DIR, 'employees')
    os.makedirs(employees_dir, exist_ok=True)
    safe_name = name.replace(' ', '_')
    # Store as encrypted .enc file (not readable as image)
    filename = f"{emp_id}_{safe_name}.enc"
    filepath = os.path.normpath(os.path.join(employees_dir, filename))

    # Read uploaded image into memory
    tmp_path = os.path.join(employees_dir, f"_tmp_{emp_id}.jpg")
    photo.save(tmp_path)
    img = cv2.imread(tmp_path)
    os.remove(tmp_path)
    if img is None:
        return jsonify({'error': 'Could not read image.'}), 400

    # Validate face present before saving
    if face_net:
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        face_found = any(detections[0, 0, i, 2] > 0.3 for i in range(detections.shape[2]))
        if not face_found:
            return jsonify({'error': 'No face detected in the photo.'}), 400

    # Encrypt and save
    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    encrypted_bytes = fernet.encrypt(img_encoded.tobytes())
    with open(filepath, 'wb') as f:
        f.write(encrypted_bytes)
    print(f'[SECURITY] Employee image encrypted and saved: {filename}')

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO employees (name, emp_id, image_path) VALUES (?, ?, ?)", (name, emp_id, filepath))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        os.remove(filepath)
        return jsonify({'error': f'Employee ID "{emp_id}" already exists.'}), 409
    if face_net and embedder_net:
        with face_data_lock:
            load_known_faces()
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
        with face_data_lock:
            load_known_faces()
    return jsonify({'success': True, 'name': emp_name})

# =============================================
#  ENTRY POINT
# =============================================

if __name__ == '__main__':
    print("=" * 50)
    print("  I.V.S.S. - Intelligent Video Surveillance")
    print("  [ Production WebSocket Architecture ]")
    print("=" * 50)
    print(f"  Face Detection  : {'ENABLED' if face_net else 'DISABLED'}")
    print(f"  Face Recognition: {'ENABLED' if embedder_net else 'DISABLED'}")
    print(f"  Weapon Detection: {'ENABLED' if yolo_net else 'DISABLED'}")
    print(f"  Camera Source   : Client Browser (WebSocket)")
    print(f"  Threading       : Per-Camera Dedicated Pairs")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
