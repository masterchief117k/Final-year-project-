# Intelligent Video Surveillance System (I.V.S.S.)

I.V.S.S. is a comprehensive, AI-powered intelligent video surveillance web application built with a Flask backend, a modern glassmorphic frontend, and real-time computer vision capabilities. 

The system runs locally and uses the host machine's webcam or connected camera feeds to provide continuous security monitoring, identify known personnel, detect threats, and manage physical access through intelligent logging.

## ✨ Features Implemented

### 1. Advanced Computer Vision Pipeline
- **Face Detection & Recognition:** Uses SSD ResNet (Caffe) to locate faces in the stream and OpenFace to compute 128-d embeddings. It recognizes registered employees and marks them for attendance.
- **Weapon Detection:** Integrated YOLOv5 object detection tailored to recognize specific threat classes like 'knife'. Immediately triggers critical UI alerts and logs the event if a threat is detected.
- **Unknown Personnel Alerts:** Generates warnings when an unrecognized face is visible on camera for an extended duration.
- **Camera Tamper Detection:** Monitors overall frame brightness and alerts the command center if the camera lens is covered or unexpectedly goes completely dark for more than 5 seconds.

### 2. Command Center Dashboard (Frontend)
- **Modern UI:** A stunning, premium dark-mode dashboard styled with glassmorphism (translucency, blur backing) and cyan/blue neon accents.
- **Real-Time Video Stream:** Live camera feed rendered smoothly via Multipart JPEG streaming (`multipart/x-mixed-replace`).
- **Activity & Attendance Logs:** A live-updating sidebar panel showing database-backed chronological events (Attendance, Security Risks, Tampering).
- **Instant Toasts & Audio:** Real-time push notifications delivered via WebSockets (`Socket.IO`). Critical threat alerts trigger an audible beep.

### 3. Employee Management System
- **Full CRUD API:** REST endpoints to seamlessly add and remove employees.
- **Dynamic Hot-Reloading:** Adding a new employee via the dashboard (with a name, employee ID, and photo) automatically updates the system's face recognition embeddings in memory without requiring a server reboot.
- **Image Storage:** Employee photos are stored locally in `/static/employees/` while metadata lives in the database.

### 4. Robust Data Management (SQLite)
The application relies on a fast, embedded SQLite database (`database.db`) maintaining four core tables:
- `users`: Administrator credentials (passwords hashed via bcrypt).
- `attendance`: Timestamps for recognized employees.
- `alerts`: Security incidents (weapons, tampering, unknown faces).
- `employees`: Registered personnel directory linking `emp_id` to their respective image paths.

---

## 🛠 Tech Stack

**Frontend:** HTML5, Vanilla JavaScript, CSS3 (Custom Glassmorphism + Keyframe Animations), Bootstrap 5 (Toasts & Modals).  
**Backend:** Python 3, Flask, Flask-SocketIO (WebSockets), Flask-Bcrypt.  
**Computer Vision:** OpenCV (`cv2.dnn`), PyTorch, YOLOv5 (Ultralytics Hub).  
**Database:** SQLite3.  

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ (ensure `python` and `pip` are on your system path).
- A working webcam.

### 1. Clone the repository
Ensure you have the project directory on your local machine.
```bash
cd Final-year-project-/project
```

### 2. Set up the Python Environment
It is highly recommended to use a virtual environment.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
The application requires several major ML and web libraries:
```bash
pip install flask flask-socketio flask-bcrypt opencv-python numpy torch ultralytics
```

### 4. Run the Application
Start the Flask server:
```bash
python app.py
```
*Note: The first time you run this, YOLOv5s will automatically download its initial PyTorch weights (`yolov5s.pt`).*

### 5. Access the Dashboard
1. Open your browser and navigate to `http://localhost:5000`.
2. Login using the default administrator credentials:
   - **Username:** `admin`
   - **Password:** `password123`
3. Ensure your browser allows camera permissions when the webcam activates.

---

## 📁 Project Structure

```text
Final-year-project-/
│
├── project/
│   ├── app.py                  # Main Flask application and Video CV loop
│   ├── database.db             # SQLite database (auto-generated)
│   ├── models/                 # Cached CV Models (ResNet SSD, OpenFace)
│   ├── static/
│   │   ├── style.css           # Core styling and glassmorphism themes
│   │   ├── script.js           # Frontend logic (Socket.IO, AJAX fetch, UI updates)
│   │   └── employees/          # Directory storing registered employee images
│   └── templates/
│       ├── login.html          # Login portal
│       └── dashboard.html      # Main operational command center
```

---

## 🤝 Adding Changes (For Collaborators)

If you are new to the project and looking to contribute, keep the following workflows in mind:

1. **Changing detection logic:** Modify the `generate_frames()` generator inside `app.py`. This loop handles reading bounding boxes, distance calculations for face embeddings, and YOLO object detection checks.
2. **Adding a new Alert Type:** If you create a new security rule in `app.py`, use the `socketio.emit('alert', ...)` function to push it to the frontend. Ensure you tag it as either `danger`, `warning`, or `success`. 
3. **Frontend Changes:** CSS is entirely held in `static/style.css`. The frontend is decoupled from the backend rendering as much as possible to allow easy updates. Modals for new forms should use Bootstrap 5 markup in `dashboard.html`.
4. **Model Swaps:** If you want to use a more accurate YOLO model or substitute OpenFace for something else (like RT-DETR), update the `# MODEL LOADING` sector at the top of `app.py`. Ensure your new model returns similar bounding-box coordinate schemas.
