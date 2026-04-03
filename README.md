# Intelligent Video Surveillance System (I.V.S.S.)

Welcome to the **I.V.S.S. Project**! 

This document is designed to explain the entire system clearly—whether you are a developer, an inspector, or someone without a deep technical background. Our goal was to build a "smart" security camera system that doesn't just record video, but actually understands what it is looking at. 

By using Artificial Intelligence (AI), this system can automatically recognize registered employees, catch intruders, and even spot dangerous weapons—all in real time.

---

## What Does This System Do?

Imagine a digital security guard that never blinks. Our system connects to a standard webcam and provides four main features:

1. **Automated Attendance:** When an employee walks in front of the camera, the system recognizes their face and automatically logs their attendance in a secure database.
2. **Stranger Alerts:** If an unknown person stands in front of the camera for too long, a warning alert is sent to the dashboard.
3. **Weapon Detection:** The system is trained to identify threats like knives. If a weapon is spotted, it immediately sounds an alarm and flashes a red alert on the screen.
4. **Camera Tampering Detection:** If an intruder tries to cover the camera lens with their hand or spray paint, the system notices that the room went unnaturally dark and alerts security.

---

## Technology Stack (Explained Simply)

To make this magic happen, we glued together several different technologies. Here is our "Tech Stack" translated into plain English:

### 1. The "Front Door" (User Interface)
*What the user sees and interacts with.*
- **HTML/CSS/JavaScript:** The building blocks of any website. We used modern design techniques (called "Glassmorphism") to make the dashboard look like a futuristic, sleek command center.
- **Bootstrap 5:** A helper tool that makes creating buttons, pop-up windows, and alerts much faster and ensures they look good on any screen.

### 2. The "Brain" (Backend Server)
*The invisible manager that processes data and connects everything.*
- **Python & Flask:** Python is our programming language, and Flask is a lightweight "server." Think of Flask as a restaurant waiter taking requests from the user interface and fetching data (like video frames or employee records) from the kitchen.
- **WebSockets (Socket.IO):** This gives us instant communication. Instead of the browser constantly asking, "Is there an alert yet?", WebSockets allow the server to instantly push an alarm to the screen the millisecond a threat is detected.

### 3. The "Eyes" (Artificial Intelligence & Computer Vision)
*How the computer understands images.*
- **OpenCV:** A digital tool that captures video from the webcam and slices it up into thousands of individual pictures (frames) every second.
- **Face Recognition (OpenFace):** When the system sees a face, this AI measures the distance between the eyes, the shape of the jaw, etc., and turns the face into a unique mathematical code. We then compare that code against our list of employee codes.
- **Weapon Detection (YOLOv5):** "YOLO" stands for *You Only Look Once*. It is an incredibly fast AI that scans the whole image instantly to look for specific shapes it was trained on—in this case, weapons.

### 4. The "Filing Cabinet" (Database)
*Where we save our records permanently.*
- **SQLite:** A small, lightweight database that lives right inside the project folder. It acts as a set of spreadsheets saving Administrator passwords securely, tracking the exact second an employee was seen, and logging every security alert.

---

## How We Built It: Our Coding Plan

To get to this final product, our team followed a structured, step-by-step coding plan. Here is the journey of how we built I.V.S.S.:

### Step 1: Laying the Foundation (The Website)
- **Goal:** Create a place to view the camera.
- **Action:** We set up the Python Flask server to host a basic webpage. We created a secure login screen so only administrators could access the camera.

### Step 2: Getting the Camera Working (The Eyes)
- **Goal:** Stream live video to the web browser.
- **Action:** We programmed Python (using OpenCV) to turn on the laptop webcam, grab the images, and stream them directly into our sleek Glassmorphism dashboard.

### Step 3: Making the System "Smart" (Face Detection)
- **Goal:** Look for humans.
- **Action:** We added a basic AI model that draws boxes around human faces. We then created an "Employee Management" panel in the dashboard allowing admins to upload a photo and name of a new employee.

### Step 4: Teaching the System to Remember (Face Recognition)
- **Goal:** Know who is who.
- **Action:** We integrated the **OpenFace** AI. Now, when the system sees a face, it compares it to the uploaded employee photos. If it matches, it prints the employee's name in green and logs their attendance in the database. If it doesn't match, it prints "Unknown" in red.

### Step 5: Adding Threat Detection (Weapons & Tampering)
- **Goal:** Catch bad guys and vandals.
- **Action:** 
  - We installed **YOLOv5**, an advanced visual AI, and plugged it into our camera stream to specifically hunt for knives.
  - We wrote a custom mathematical rule: *If the average brightness of the image drops to near-zero suddenly, assume the camera has been covered/tampered with.*

### Step 6: Real-Time Communication (Making it "Pop")
- **Goal:** Security guards need to be warned instantly.
- **Action:** We added **WebSockets** and an audio buzzer. Now, the moment a weapon or unknown person is seen, the Python backend shouts to the web browser, which instantly flashes a red notification on the screen and plays a warning beep!

---

## How to Run the Project

If you are inspecting this project and want to run it yourself, simply follow these basic commands in your terminal:

**1. Open your terminal and go into the project folder:**
```bash
cd Final-year-project-/project
```

**2. Turn on the "Virtual Environment"**
*(This isolates our project's tools from the rest of your computer)*
```bash
# On Windows
.\venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

**3. Install the required tools:**
```bash
pip install flask flask-socketio flask-bcrypt opencv-python numpy torch ultralytics
```

**4. Start the Application:**
```bash
python app.py
```

**5. View the System:**
Open Google Chrome (or any browser) and type `http://localhost:5000` into the search bar. Log in using the default admin credentials provided by the team!
