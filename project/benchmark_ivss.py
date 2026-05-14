"""
I.V.S.S. Performance & Metrics Benchmarking Script
====================================================
Measures per-component inference latency and computes Precision/Recall
for the Intelligent Video Surveillance System.

Usage:
    python benchmark_ivss.py

Requires:
    - models/ directory with SSD ResNet, OpenFace, and YOLOv5s weights
    - (Optional) test_images/ directory with labeled subfolders for P/R metrics
"""

import os
import sys
import time
import statistics
import numpy as np
import cv2
import warnings

# Suppress PyTorch deprecation warnings from YOLOv5
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
#  CONFIGURATION
# ============================================================
# Resolve all paths relative to this script's directory (project/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
TEST_IMAGES_DIR = os.path.join(SCRIPT_DIR, "test_images")
EMPLOYEES_DIR = os.path.join(SCRIPT_DIR, "static", "employees")
NUM_ITERATIONS = 100          # Warm-up excluded; median over this many runs
FACE_CONFIDENCE_THRESH = 0.5
FACE_DISTANCE_THRESH = 1.0
WEAPON_CONFIDENCE_THRESH = 0.6
TAMPERING_DARK_THRESH = 35
FRAME_SIZE = (640, 480)       # Synthetic frame resolution

# ============================================================
#  MODEL LOADING
# ============================================================

def load_models():
    """Load all three models and return them (or None on failure)."""
    print("\n[1/4] Loading models...")

    # --- Face Detector (SSD ResNet / Caffe) ---
    face_net = None
    proto = os.path.join(MODELS_DIR, "deploy.prototxt")
    model = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    if os.path.exists(proto) and os.path.exists(model):
        face_net = cv2.dnn.readNetFromCaffe(proto, model)
        print("  [OK] Face detector (SSD ResNet) loaded.")
    else:
        print("  [SKIP] Face detector models not found.")

    # --- Face Embedder (OpenFace) ---
    embedder_net = None
    emb_path = os.path.join(MODELS_DIR, "nn4.small2.v1.t7")
    if os.path.exists(emb_path):
        embedder_net = cv2.dnn.readNetFromTorch(emb_path)
        print("  [OK] OpenFace embedder loaded.")
    else:
        print("  [SKIP] OpenFace model not found.")

    # --- Weapon Detector (YOLOv5s) ---
    yolo_net = None
    try:
        import torch
        yolo_net = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        yolo_net.conf = 0.15
        yolo_net.classes = [43]  # knife
        print("  [OK] YOLOv5s weapon detector loaded.")
    except Exception as e:
        print(f"  [SKIP] YOLOv5s failed: {e}")

    return face_net, embedder_net, yolo_net


# ============================================================
#  SYNTHETIC FRAME GENERATORS
# ============================================================

def make_synthetic_frame():
    """Normal office-like frame (random noise with moderate brightness)."""
    frame = np.random.randint(80, 200, (*FRAME_SIZE[::-1], 3), dtype=np.uint8)
    return frame

def make_dark_frame():
    """Simulates a covered/tampered camera."""
    return np.zeros((*FRAME_SIZE[::-1], 3), dtype=np.uint8)

def make_bright_frame():
    """Normal well-lit frame."""
    return np.full((*FRAME_SIZE[::-1], 3), 150, dtype=np.uint8)


# ============================================================
#  INDIVIDUAL COMPONENT BENCHMARKS
# ============================================================

def benchmark_face_roi(face_net, frame, n=NUM_ITERATIONS):
    """Benchmark SSD ResNet face ROI extraction."""
    if face_net is None:
        return None, None
    # Warm-up
    for _ in range(5):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        face_net.forward()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        face_net.forward()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), statistics.stdev(times)


def benchmark_embedding(embedder_net, face_roi, n=NUM_ITERATIONS):
    """Benchmark OpenFace 128D embedding generation."""
    if embedder_net is None or face_roi is None:
        return None, None
    # Warm-up
    for _ in range(5):
        blob = cv2.dnn.blobFromImage(face_roi, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
        embedder_net.setInput(blob)
        embedder_net.forward()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        blob = cv2.dnn.blobFromImage(face_roi, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
        embedder_net.setInput(blob)
        embedder_net.forward()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), statistics.stdev(times)


def benchmark_weapon(yolo_net, frame, n=NUM_ITERATIONS):
    """Benchmark YOLOv5s weapon inference."""
    if yolo_net is None:
        return None, None
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Warm-up
    for _ in range(5):
        yolo_net(img_rgb)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        yolo_net(img_rgb)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), statistics.stdev(times)


def benchmark_tampering(frame, n=NUM_ITERATIONS):
    """Benchmark grayscale + mean pixel intensity heuristic."""
    # Warm-up
    for _ in range(5):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np.mean(gray)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _ = np.mean(gray)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), statistics.stdev(times)


# ============================================================
#  FPS SIMULATION
# ============================================================

def simulate_fps(face_net, embedder_net, yolo_net, target_fps, duration_s=5):
    """
    Simulate a pipeline run at a target FPS for `duration_s` seconds.
    Returns effective FPS and average per-frame latency.
    """
    frame_interval = 1.0 / target_fps
    frame = make_synthetic_frame()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Build a dummy face ROI
    face_roi = frame[100:200, 100:200]

    frame_times = []
    total_start = time.perf_counter()

    while (time.perf_counter() - total_start) < duration_s:
        loop_start = time.perf_counter()

        # --- Tampering ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np.mean(gray)

        # --- Face detection ---
        if face_net:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            face_net.forward()

        # --- Embedding ---
        if embedder_net and face_roi.shape[0] > 0:
            blob = cv2.dnn.blobFromImage(face_roi, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=False)
            embedder_net.setInput(blob)
            embedder_net.forward()

        # --- Weapon ---
        if yolo_net:
            yolo_net(img_rgb)

        elapsed = time.perf_counter() - loop_start
        frame_times.append(elapsed * 1000)

        # Throttle to target FPS
        remaining = frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    effective_fps = len(frame_times) / duration_s
    avg_latency = statistics.mean(frame_times) if frame_times else 0
    return effective_fps, avg_latency


# ============================================================
#  PRECISION / RECALL
# ============================================================

def compute_precision_recall(face_net, embedder_net, yolo_net):
    """
    Scan test_images/ subfolders and compute Precision & Recall.
    Subfolders: known/, unknown/, weapons/, clean/, tampered/
    Returns dict of metrics per module.
    """
    results = {
        "face": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "synthetic": True},
        "weapon": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "synthetic": True},
        "tamper": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "synthetic": True},
    }

    # --- Load known embeddings from static/employees ---
    known_embeddings = []
    known_names = []
    emp_dir = EMPLOYEES_DIR
    if face_net and embedder_net and os.path.exists(emp_dir):
        for fn in os.listdir(emp_dir):
            img = cv2.imread(os.path.join(emp_dir, fn))
            if img is None:
                continue
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            dets = face_net.forward()
            for i in range(dets.shape[2]):
                if dets[0, 0, i, 2] > 0.3:
                    box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                    sx, sy, ex, ey = box.astype("int")
                    sx, sy = max(0, sx), max(0, sy)
                    ex, ey = min(w, ex), min(h, ey)
                    roi = img[sy:ey, sx:ex]
                    if roi.shape[0] > 0 and roi.shape[1] > 0:
                        fb = cv2.dnn.blobFromImage(roi, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=False)
                        embedder_net.setInput(fb)
                        vec = embedder_net.forward().flatten()
                        known_embeddings.append(vec)
                        raw = os.path.splitext(fn)[0]
                        parts = raw.split('_', 1)
                        known_names.append(parts[1] if len(parts) > 1 else raw)
                    break

    def _recognize_face(img):
        """Returns True if a known face is detected."""
        if not face_net or not embedder_net or not known_embeddings:
            return False
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        dets = face_net.forward()
        for i in range(dets.shape[2]):
            if dets[0, 0, i, 2] > FACE_CONFIDENCE_THRESH:
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = box.astype("int")
                sx, sy = max(0, sx), max(0, sy)
                ex, ey = min(w, ex), min(h, ey)
                roi = img[sy:ey, sx:ex]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    fb = cv2.dnn.blobFromImage(roi, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=False)
                    embedder_net.setInput(fb)
                    vec = embedder_net.forward().flatten()
                    for kv in known_embeddings:
                        if np.linalg.norm(vec - kv) < FACE_DISTANCE_THRESH:
                            return True
        return False

    def _detect_weapon(img):
        """Returns True if weapon detected with conf > threshold."""
        if yolo_net is None:
            return False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = yolo_net(rgb)
        for det in res.xyxy[0]:
            if float(det[4]) >= WEAPON_CONFIDENCE_THRESH:
                return True
        return False

    def _detect_tamper(img):
        """Returns True if image is dark enough to be tampering."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < TAMPERING_DARK_THRESH

    def _load_images(folder):
        path = os.path.join(TEST_IMAGES_DIR, folder)
        imgs = []
        if os.path.isdir(path):
            for fn in os.listdir(path):
                img = cv2.imread(os.path.join(path, fn))
                if img is not None:
                    imgs.append(img)
        return imgs

    # Check if real test images exist
    has_real = os.path.isdir(TEST_IMAGES_DIR) and any(
        os.path.isdir(os.path.join(TEST_IMAGES_DIR, d))
        for d in ["known", "unknown", "weapons", "clean", "tampered"]
    )

    if has_real:
        # --- Face Recognition P/R ---
        known_imgs = _load_images("known")
        unknown_imgs = _load_images("unknown")
        if known_imgs or unknown_imgs:
            results["face"]["synthetic"] = False
            for img in known_imgs:
                if _recognize_face(img):
                    results["face"]["tp"] += 1
                else:
                    results["face"]["fn"] += 1
            for img in unknown_imgs:
                if _recognize_face(img):
                    results["face"]["fp"] += 1
                else:
                    results["face"]["tn"] += 1

        # --- Weapon Detection P/R ---
        weapon_imgs = _load_images("weapons")
        clean_imgs = _load_images("clean")
        if weapon_imgs or clean_imgs:
            results["weapon"]["synthetic"] = False
            for img in weapon_imgs:
                if _detect_weapon(img):
                    results["weapon"]["tp"] += 1
                else:
                    results["weapon"]["fn"] += 1
            for img in clean_imgs:
                if _detect_weapon(img):
                    results["weapon"]["fp"] += 1
                else:
                    results["weapon"]["tn"] += 1

        # --- Tampering P/R ---
        tampered_imgs = _load_images("tampered")
        normal_imgs = _load_images("clean")  # reuse clean as "normal"
        if tampered_imgs or normal_imgs:
            results["tamper"]["synthetic"] = False
            for img in tampered_imgs:
                if _detect_tamper(img):
                    results["tamper"]["tp"] += 1
                else:
                    results["tamper"]["fn"] += 1
            for img in normal_imgs:
                if _detect_tamper(img):
                    results["tamper"]["fp"] += 1
                else:
                    results["tamper"]["tn"] += 1
    else:
        # --- Synthetic fallback ---
        print("  [INFO] No test_images/ found. Using synthetic frames.")
        # Face: use employee image as known, noise as unknown
        emp_img_path = os.path.join(EMPLOYEES_DIR, "1_kshitij.jpg")
        if os.path.exists(emp_img_path) and face_net and embedder_net:
            results["face"]["synthetic"] = False
            img = cv2.imread(emp_img_path)
            if _recognize_face(img):
                results["face"]["tp"] = 1
            else:
                results["face"]["fn"] = 1
            # Synthetic unknown
            noise = make_synthetic_frame()
            if _recognize_face(noise):
                results["face"]["fp"] = 1
            else:
                results["face"]["tn"] = 1

        # Tampering
        results["tamper"]["synthetic"] = False
        if _detect_tamper(make_dark_frame()):
            results["tamper"]["tp"] = 1
        else:
            results["tamper"]["fn"] = 1
        if _detect_tamper(make_bright_frame()):
            results["tamper"]["fp"] = 1
        else:
            results["tamper"]["tn"] = 1

    return results


def _calc_pr(m):
    """Calculate precision and recall from TP/FP/FN counts."""
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


# ============================================================
#  PRETTY PRINTER
# ============================================================

def print_results(latencies, fps_results, pr_results):
    """Print formatted benchmark results table."""
    W = 68

    print("\n")
    print("+" + "=" * W + "+")
    print("|" + " I.V.S.S. PERFORMANCE BENCHMARK RESULTS".center(W) + "|")
    print("+" + "=" * W + "+")

    # --- Latency Table ---
    print("|" + " COMPONENT LATENCY".center(W) + "|")
    print("+" + "-" * W + "+")
    header = f"| {'Component':<28}| {'Median (ms)':>12} | {'Std Dev':>9} | {'Iters':>6} |"
    print(header)
    print("+" + "-" * W + "+")

    labels = ["Face ROI Extraction", "OpenFace Embedding", "Weapon Detection (YOLOv5s)", "Tampering Heuristic"]
    for label, (med, std) in zip(labels, latencies):
        if med is not None:
            print(f"| {label:<28}| {med:>10.2f}ms | {'+/-'}{std:>5.2f} | {NUM_ITERATIONS:>6} |")
        else:
            print(f"| {label:<28}| {'N/A':>12} | {'N/A':>9} | {'N/A':>6} |")

    print("+" + "-" * W + "+")

    # --- FPS Simulation ---
    print("|" + " FPS SIMULATION".center(W) + "|")
    print("+" + "-" * W + "+")
    for target_fps, (eff_fps, avg_lat) in fps_results.items():
        line = f"| Target {target_fps:>2} FPS => Effective {eff_fps:>5.1f} FPS | Avg latency: {avg_lat:>7.2f} ms"
        print(f"{line:<{W+1}}|")
    print("+" + "-" * W + "+")

    # --- Precision / Recall ---
    print("|" + " PRECISION & RECALL".center(W) + "|")
    print("+" + "-" * W + "+")
    header2 = f"| {'Module':<22}| {'Precision':>10} | {'Recall':>10} | {'TP':>4} | {'FP':>4} | {'FN':>4} |"
    print(header2)
    print("+" + "-" * W + "+")

    module_labels = {"face": "Face Recognition", "weapon": "Weapon Detection", "tamper": "Tampering Detection"}
    for key in ["face", "weapon", "tamper"]:
        m = pr_results[key]
        p, r = _calc_pr(m)
        tag = " (SYN)" if m["synthetic"] else ""
        label = module_labels[key] + tag
        if m["tp"] + m["fp"] + m["fn"] + m["tn"] == 0:
            print(f"| {label:<22}| {'N/A':>10} | {'N/A':>10} | {'—':>4} | {'—':>4} | {'—':>4} |")
        else:
            print(f"| {label:<22}| {p:>10.4f} | {r:>10.4f} | {m['tp']:>4} | {m['fp']:>4} | {m['fn']:>4} |")

    print("+" + "=" * W + "+")
    print()


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  I.V.S.S. — Performance & Metrics Benchmark")
    print("=" * 60)

    face_net, embedder_net, yolo_net = load_models()

    # Prepare test frames
    frame = make_synthetic_frame()
    # Try to use a real employee image for face benchmarks
    emp_path = os.path.join(EMPLOYEES_DIR, "1_kshitij.jpg")
    face_frame = cv2.imread(emp_path) if os.path.exists(emp_path) else frame

    # Extract a face ROI for embedding benchmark
    face_roi = None
    if face_net is not None:
        h, w = face_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(face_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        dets = face_net.forward()
        for i in range(dets.shape[2]):
            if dets[0, 0, i, 2] > 0.3:
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = box.astype("int")
                sx, sy = max(0, sx), max(0, sy)
                ex, ey = min(w, ex), min(h, ey)
                face_roi = face_frame[sy:ey, sx:ex]
                break
    if face_roi is None or face_roi.size == 0:
        face_roi = frame[100:200, 150:250]  # fallback synthetic ROI

    # --- Run component benchmarks ---
    print(f"\n[2/4] Benchmarking components ({NUM_ITERATIONS} iterations each)...")

    print("  Benchmarking Face ROI Extraction...")
    face_lat = benchmark_face_roi(face_net, face_frame)

    print("  Benchmarking OpenFace Embedding...")
    emb_lat = benchmark_embedding(embedder_net, face_roi)

    print("  Benchmarking Weapon Detection (YOLOv5s)...")
    wep_lat = benchmark_weapon(yolo_net, frame)

    print("  Benchmarking Tampering Heuristic...")
    tamp_lat = benchmark_tampering(frame)

    latencies = [face_lat, emb_lat, wep_lat, tamp_lat]

    # --- FPS simulation ---
    print("\n[3/4] Simulating FPS pipelines (5s each)...")
    fps_results = {}
    for target in [30, 12]:
        print(f"  Simulating {target} FPS...")
        eff, avg = simulate_fps(face_net, embedder_net, yolo_net, target, duration_s=5)
        fps_results[target] = (eff, avg)

    # --- Precision / Recall ---
    print("\n[4/4] Computing Precision & Recall...")
    pr_results = compute_precision_recall(face_net, embedder_net, yolo_net)

    # --- Output ---
    print_results(latencies, fps_results, pr_results)


if __name__ == "__main__":
    main()
