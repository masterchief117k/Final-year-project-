import os
import cv2
import time
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, Response, session, jsonify
from flask_socketio import SocketIO, emit
import sqlite3
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.secret_key = "ivss_super_secret"
socketio = SocketIO(app)
bcrypt = Bcrypt(app)

# --- Database Setup ---
DB_NAME = "database.db"

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
model_path = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
config_path = os.path.join("models", "deploy.prototxt")
if os.path.exists(model_path) and os.path.exists(config_path):
    face_net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    print("[INFO] Face detection model loaded.")
else:
    face_net = None
    print("[WARNING] Face detection models not found in models/ directory!")

# --- 2. Face Embedding Model (OpenFace) ---
embedder_path = os.path.join("models", "openface_nn4.small2.v1.t7")
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
    
    employees_dir = os.path.join("static", "employees")
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

# --- Webcam Setup ---
camera = cv2.VideoCapture(0)

# =============================================
#  STATE VARIABLES
# =============================================
last_face_alert_time = 0
last_weapon_alert_time = 0
last_camera_blocked_alert_time = 0

FACE_ALERT_COOLDOWN = 5       # seconds between face alerts
WEAPON_ALERT_COOLDOWN = 5     # seconds between weapon alerts
CAMERA_BLOCKED_COOLDOWN = 15  # seconds between camera-blocked alerts

# Camera blocked tracking
camera_dark_start = None       # timestamp when frame first went dark
CAMERA_DARK_THRESHOLD = 35     # raised from 15 to 35: most webcams output 20-40 noise when covered
CAMERA_BLOCKED_DURATION = 5    # reduced from 10 to 5 seconds of continuous darkness before alert

# YOLO inference throttle (run every N frames to save CPU)
YOLO_FRAME_INTERVAL = 3
frame_counter = 0
last_yolo_results = []  # cache YOLO results between intervals

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

    employees_dir = os.path.join('static', 'employees')
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
#  VIDEO PROCESSING PIPELINE
# =============================================
def generate_frames():
    global last_face_alert_time, last_weapon_alert_time, last_camera_blocked_alert_time
    global camera_dark_start, frame_counter, last_yolo_results
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        h, w = frame.shape[:2]
        current_time = time.time()
        frame_counter += 1
        
        # -----------------------------------------------
        #  FEATURE 1: Camera Blocked Detection
        # -----------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < CAMERA_DARK_THRESHOLD:
            # Frame is very dark
            if camera_dark_start is None:
                camera_dark_start = current_time  # start tracking
            
            dark_duration = current_time - camera_dark_start
            
            if dark_duration >= CAMERA_BLOCKED_DURATION:
                # Camera has been blocked for too long!
                # Draw warning on frame
                cv2.putText(frame, "!! CAMERA BLOCKED !!", (w // 2 - 200, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, f"Blocked for {dark_duration:.0f}s", (w // 2 - 120, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Send alert (throttled)
                if (current_time - last_camera_blocked_alert_time) > CAMERA_BLOCKED_COOLDOWN:
                    socketio.emit('alert', {
                        'message': f'CAMERA BLOCKED! Feed has been dark for {dark_duration:.0f} seconds. Possible tampering!',
                        'type': 'warning'
                    }, namespace='/')
                    log_alert_to_db("Tampering", f"Camera blocked for {dark_duration:.0f}s")
                    last_camera_blocked_alert_time = current_time
                    print(f"[ALERT] Camera blocked for {dark_duration:.0f}s!")
        else:
            # Frame is bright enough — reset dark tracker
            camera_dark_start = None
            
        # Debug brightness every N frames
        if frame_counter % 30 == 0:
            print(f"[CAM] Brightness: {mean_brightness:.1f}")
        
        # -----------------------------------------------
        #  FEATURE 2: Face Detection & Recognition
        # -----------------------------------------------
        if face_net:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)
                    
                    face_roi = frame[startY:endY, startX:endX]
                    
                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0 and embedder_net:
                        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        embedder_net.setInput(face_blob)
                        vec = embedder_net.forward().flatten()
                        
                        min_dist = float("inf")
                        best_match = None
                        
                        for j, known_vec in enumerate(known_face_embeddings):
                            dist = np.linalg.norm(vec - known_vec)
                            if dist < min_dist:
                                min_dist = dist
                                best_match = known_face_names[j]
                                
                        # Debug: print distance every 30 frames
                        if frame_counter % 30 == 0 and best_match is not None:
                            print(f"[FACE] Closest match: '{best_match}' dist={min_dist:.3f} (threshold=1.0)")
                        
                        if min_dist < 1.0 and best_match is not None:
                            name = best_match
                            color = (0, 255, 0)  # Green for known
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    label = f"{name} ({confidence*100:.0f}%)"
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Throttled Alerts
                    if (current_time - last_face_alert_time) > FACE_ALERT_COOLDOWN:
                        if name == "Unknown":
                            socketio.emit('alert', {
                                'message': 'UNKNOWN FACE detected in camera feed!',
                                'type': 'danger'
                            }, namespace='/')
                            log_alert_to_db("Security", "Unknown face detected")
                        else:
                            socketio.emit('alert', {
                                'message': f'Attendance logged for {name}',
                                'type': 'success'
                            }, namespace='/')
                            conn = sqlite3.connect(DB_NAME)
                            c = conn.cursor()
                            c.execute("INSERT INTO attendance (name) VALUES (?)", (name,))
                            conn.commit()
                            conn.close()
                            
                        last_face_alert_time = current_time
        
        # -----------------------------------------------
        #  FEATURE 3: Weapon Detection (YOLOv5)
        # -----------------------------------------------
        weapon_found = False
        
        if yolo_net:
            if frame_counter % YOLO_FRAME_INTERVAL == 0:
                # Convert BGR to RGB for YOLOv5
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference
                results = yolo_net(img_rgb)
                
                weapon_detections = []
                # parse detections (xyxy format: x1, y1, x2, y2, confidence, class)
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det
                    conf = float(conf)
                    
                    if conf >= WEAPON_CONFIDENCE:
                        # YOLOv5 returns class names via names attribute
                        class_name = yolo_net.names[int(cls)]
                        weapon_detections.append({
                            'class': class_name.upper(),
                            'confidence': conf,
                            'box': (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        })
                        print(f"[YOLOv5] {class_name.upper()} ({conf:.2f})", flush=True)
                
                last_yolo_results = weapon_detections
        
        # --- Draw weapon detections ---
        if last_yolo_results:
            weapon_found = True
            # Red flashing border
            border_alpha = int(abs(np.sin(time.time() * 5)) * 255)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, border_alpha), 4)
            
            for det in last_yolo_results:
                x, y, bw, bh = det['box']
                conf = det['confidence']
                cls = det['class']
                
                # Draw red warning box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                
                # Warning label with background
                weapon_label = f"WEAPON: {cls} ({conf*100:.0f}%)"
                (label_w, label_h), _ = cv2.getTextSize(weapon_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 5, y), (0, 0, 200), -1)
                cv2.putText(frame, weapon_label, (x + 2, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # --- Trigger weapon alert (throttled) ---
        if weapon_found and (current_time - last_weapon_alert_time) > WEAPON_ALERT_COOLDOWN:
            weapon_names = list(set([d['class'] for d in last_yolo_results]))
            weapon_str = ", ".join(weapon_names)
            socketio.emit('alert', {
                'message': f'WEAPON DETECTED: {weapon_str}! Immediate action required!',
                'type': 'danger'
            }, namespace='/')
            log_alert_to_db("Weapon", f"Weapon detected: {weapon_str}")
            last_weapon_alert_time = current_time
            print(f"[ALERT] Weapon detected: {weapon_str}", flush=True)

        # Encode frame for stream
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
    print("=" * 50)
    print(f"  Face Detection : {'ENABLED' if face_net else 'DISABLED'}")
    print(f"  Face Recognition: {'ENABLED' if embedder_net else 'DISABLED'}")
    print(f"  Weapon Detection: {'ENABLED' if yolo_net else 'DISABLED'}")
    print(f"  Camera Tamper   : ENABLED")
    print("=" * 50)
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
