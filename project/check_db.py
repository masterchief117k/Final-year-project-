import cv2
import numpy as np
import os
import sqlite3

results = []
results.append("=" * 60)
results.append("  FACE RECOGNITION DIAGNOSTIC")
results.append("=" * 60)

# 1. Check model files
embedder_path = os.path.join("models", "nn4.small2.v1.t7")
face_model = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
face_config = os.path.join("models", "deploy.prototxt")

results.append(f"\n[1] Model files:")
results.append(f"  Embedder exists: {os.path.exists(embedder_path)}, size: {os.path.getsize(embedder_path) if os.path.exists(embedder_path) else 'N/A'} bytes")
results.append(f"  Face detector exists: {os.path.exists(face_model)}")

# 2. Try loading embedder
try:
    embedder_net = cv2.dnn.readNetFromTorch(embedder_path)
    results.append(f"\n[2] Embedder loaded: SUCCESS")
except Exception as e:
    results.append(f"\n[2] Embedder loaded: FAILED - {e}")
    with open("diagnostic_output.txt", "w") as f:
        f.write("\n".join(results))
    exit(1)

# 3. Load face detector
face_net = cv2.dnn.readNetFromCaffe(face_config, face_model)
results.append(f"[3] Face detector loaded: SUCCESS")

# 4. Check employee images
emp_dir = os.path.join("static", "employees")
files = os.listdir(emp_dir) if os.path.exists(emp_dir) else []
results.append(f"\n[4] Employee images in '{emp_dir}': {len(files)} file(s)")

for f in files:
    fpath = os.path.join(emp_dir, f)
    img = cv2.imread(fpath)
    if img is None:
        results.append(f"  X {f} - cv2.imread FAILED")
        continue
    
    h, w = img.shape[:2]
    results.append(f"  {f} ({os.path.getsize(fpath)} bytes, {w}x{h})")
    
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    best_conf = 0
    best_box = None
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > best_conf:
            best_conf = conf
            best_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    
    results.append(f"    Best face confidence: {best_conf:.4f}")
    
    if best_conf < 0.3:
        results.append(f"    NO FACE DETECTED (below 0.3 threshold)")
        continue
    
    (startX, startY, endX, endY) = best_box.astype("int")
    startX, startY = max(0, startX), max(0, startY)
    endX, endY = min(w, endX), min(h, endY)
    face_roi = img[startY:endY, startX:endX]
    results.append(f"    Face ROI: {face_roi.shape}")
    
    face_blob = cv2.dnn.blobFromImage(face_roi, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
    embedder_net.setInput(face_blob)
    vec = embedder_net.forward().flatten()
    results.append(f"    Embedding OK! norm={np.linalg.norm(vec):.4f}, first3={vec[:3]}")

# 5. Check DB
results.append(f"\n[5] Database:")
conn = sqlite3.connect("database.db")
c = conn.cursor()
c.execute("SELECT name, emp_id, image_path FROM employees")
rows = c.fetchall()
results.append(f"  Employees in DB: {len(rows)}")
for r in rows:
    np_path = os.path.normpath(r[2])
    results.append(f"    name='{r[0]}', id='{r[1]}', path='{r[2]}', norm='{np_path}', exists={os.path.exists(np_path)}")
conn.close()

# 6. Check what load_known_faces would do with path matching
results.append(f"\n[6] Path matching test:")
conn = sqlite3.connect("database.db")
c = conn.cursor()
c.execute("SELECT name, image_path FROM employees")
name_map = {}
for row in c.fetchall():
    name_map[os.path.normpath(row[1])] = row[0]
conn.close()
results.append(f"  name_map keys: {list(name_map.keys())}")

if os.path.exists(emp_dir):
    for f in os.listdir(emp_dir):
        fp = os.path.normpath(os.path.join(emp_dir, f))
        match = fp in name_map
        results.append(f"  File '{fp}' -> DB match: {match}")

results.append("\n" + "=" * 60)

with open("diagnostic_output.txt", "w") as f:
    f.write("\n".join(results))
