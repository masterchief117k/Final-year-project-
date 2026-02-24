from flask import Flask, render_template, request, redirect, url_for, Response, session
from flask_socketio import SocketIO, emit
import cv2

app = Flask(__name__)
app.secret_key = "supersecretkey"   # replace with env variable in production
socketio = SocketIO(app)

# --- Webcam setup ---
camera = cv2.VideoCapture(0)  # 0 = default laptop webcam

# --- Dummy user database (replace with SQLite later) ---
users = {"admin": "password123"}  # store hashed passwords in real app

# --- Routes ---
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    if username in users and users[username] == password:
        session['user'] = username
        return redirect(url_for('dashboard'))
    return "Invalid credentials", 401

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# --- Video feed route ---
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Example alert trigger ---
@app.route('/trigger_alert')
def trigger_alert():
    socketio.emit('alert', {'message': 'Weapon detected!'})
    return "Alert triggered!"

if __name__ == '__main__':
    socketio.run(app, debug=True)
