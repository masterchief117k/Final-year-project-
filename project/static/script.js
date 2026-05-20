// =============================================
//  I.V.S.S. — Production Client-Side Engine
//  Handles: camera access, frame streaming,
//  inference overlay rendering, alerts, employees
// =============================================

const socket = io();

// --- Camera State ---
let activeStreams = {};       // camera_id -> { video, stream, intervalId, canvasCtx }
let inferenceState = {};     // camera_id -> { faces, weapons, tampering, fps } — merged from both threads
let availableCameras = [];
const STREAM_FPS = 12;
const JPEG_QUALITY = 0.65;

// =============================================
//  CAMERA ENUMERATION & SELECTION
// =============================================

async function enumerateCameras() {
  try {
    // Need a temporary stream to trigger permission prompt first
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
    tempStream.getTracks().forEach(t => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    availableCameras = devices.filter(d => d.kind === 'videoinput');

    const selector = document.getElementById('cameraSelector');
    if (!selector) return;

    selector.innerHTML = '';

    availableCameras.forEach((cam, idx) => {
      const opt = document.createElement('option');
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Camera ${idx + 1}`;
      selector.appendChild(opt);
    });

    // Add "All Cameras" option if multiple exist
    if (availableCameras.length > 1) {
      const allOpt = document.createElement('option');
      allOpt.value = '__all__';
      allOpt.textContent = `All Cameras (${availableCameras.length})`;
      selector.appendChild(allOpt);
    }

    updateCameraStatus(`${availableCameras.length} camera(s) detected`);
  } catch (err) {
    console.error('Camera enumeration failed:', err);
    updateCameraStatus('Camera access denied');
  }
}

function updateCameraStatus(msg) {
  const el = document.getElementById('cameraStatus');
  if (el) el.textContent = msg;
}

// =============================================
//  CAMERA STREAM MANAGEMENT
// =============================================

async function startSelectedCamera() {
  const selector = document.getElementById('cameraSelector');
  if (!selector) return;

  // Stop all existing streams first
  stopAllStreams();

  const value = selector.value;
  if (value === '__all__') {
    // Start all cameras
    for (const cam of availableCameras) {
      await startCameraStream(cam.deviceId, cam.label || cam.deviceId);
    }
  } else {
    const cam = availableCameras.find(c => c.deviceId === value);
    await startCameraStream(value, cam ? (cam.label || 'Camera') : 'Camera');
  }

  // Notify server which cameras are active
  socket.emit('cameras_active', {
    camera_ids: Object.keys(activeStreams)
  });
}

async function startCameraStream(deviceId, label) {
  // Clear placeholder message if present
  const container = document.getElementById('videoContainer');
  const placeholder = container.querySelector('.no-feed-msg');
  if (placeholder) placeholder.remove();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: deviceId },
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    });

    // Create video element (hidden — used only for frame capture)
    const video = document.createElement('video');
    video.srcObject = stream;
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
    await video.play();

    // Create visible canvas for display + overlay
    const wrapper = document.createElement('div');
    wrapper.className = 'camera-feed-wrapper';
    wrapper.id = `cam-wrapper-${deviceId}`;
    wrapper.innerHTML = `
      <div class="cam-label">${label}</div>
      <canvas class="camera-canvas" id="canvas-${deviceId}"></canvas>
      <div class="camera-overlay">
        <div class="crosshair top-left"></div>
        <div class="crosshair top-right"></div>
        <div class="crosshair bottom-left"></div>
        <div class="crosshair bottom-right"></div>
        <div class="cam-info" id="camInfo-${deviceId}">FPS: -- | Waiting...</div>
      </div>
    `;
    container.appendChild(wrapper);

    const canvas = document.getElementById(`canvas-${deviceId}`);
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');

    // Hidden capture canvas (for encoding frames to send)
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = 640;
    captureCanvas.height = 480;
    const captureCtx = captureCanvas.getContext('2d');

    // Start frame capture interval
    const intervalId = setInterval(() => {
      if (video.readyState >= video.HAVE_CURRENT_DATA) {
        // Draw raw video to visible canvas (will be overlaid later)
        ctx.drawImage(video, 0, 0, 640, 480);

        // Encode frame and send to server
        captureCtx.drawImage(video, 0, 0, 640, 480);
        const dataUrl = captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
        socket.emit('video_frame', {
          image: dataUrl,
          camera_id: deviceId,
          timestamp: Date.now()
        });
      }
    }, 1000 / STREAM_FPS);

    activeStreams[deviceId] = {
      video, stream, intervalId, ctx, canvas, label
    };

    // Initialize per-camera inference state cache
    inferenceState[deviceId] = { faces: [], weapons: [], tampering: false, fps: 0 };

    updateCameraStatus(`Streaming ${Object.keys(activeStreams).length} camera(s)`);
  } catch (err) {
    console.error(`Failed to start camera ${deviceId}:`, err);
  }
}

function stopAllStreams() {
  for (const [camId, data] of Object.entries(activeStreams)) {
    clearInterval(data.intervalId);
    data.stream.getTracks().forEach(t => t.stop());
    const wrapper = document.getElementById(`cam-wrapper-${camId}`);
    if (wrapper) wrapper.remove();
    delete inferenceState[camId];
  }
  activeStreams = {};
  socket.emit('cameras_active', { camera_ids: [] });
}

// =============================================
//  INFERENCE RESULTS — OVERLAY RENDERING
// =============================================

socket.on('inference_results', (data) => {
  const camId = data.camera_id;
  const camData = activeStreams[camId];
  if (!camData) return;

  // --- MERGE results: only update fields that this thread actually sent ---
  if (!inferenceState[camId]) {
    inferenceState[camId] = { faces: [], weapons: [], tampering: false, fps: 0 };
  }
  const state = inferenceState[camId];

  // Thread A sends faces + tampering; Thread B sends weapons.
  // Only overwrite the field if the incoming data is non-empty OR if it's the thread that owns it.
  if (data.faces && data.faces.length > 0) {
    state.faces = data.faces;
  } else if (data.faces && data.faces.length === 0 && (!data.weapons || data.weapons.length === 0)) {
    // This is a face-thread emission with no faces detected — clear faces
    state.faces = [];
  }

  if (data.weapons && data.weapons.length > 0) {
    state.weapons = data.weapons;
    state._weaponTime = Date.now();
  }
  // Auto-clear weapons after 2 seconds of no weapon updates
  if (state._weaponTime && (Date.now() - state._weaponTime) > 2000) {
    state.weapons = [];
  }

  if (data.tampering !== undefined) state.tampering = data.tampering;
  if (data.fps !== undefined) state.fps = data.fps;

  // --- RENDER merged state ---
  const ctx = camData.ctx;
  const video = camData.video;

  // Redraw raw video frame (clears canvas)
  if (video.readyState >= video.HAVE_CURRENT_DATA) {
    ctx.drawImage(video, 0, 0, 640, 480);
  }

  // Draw ALL faces from merged state
  for (const face of state.faces) {
    const [sx, sy, ex, ey] = face.box;
    const color = face.known ? '#00ff41' : '#ff3b30';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(sx, sy, ex - sx, ey - sy);
    const label = `${face.name} (${Math.round(face.confidence * 100)}%)`;
    ctx.font = '13px Inter, sans-serif';
    const metrics = ctx.measureText(label);
    ctx.fillStyle = color;
    ctx.fillRect(sx, sy - 20, metrics.width + 10, 20);
    ctx.fillStyle = '#000';
    ctx.fillText(label, sx + 5, sy - 5);
  }

  // Draw ALL weapons from merged state
  if (state.weapons.length > 0) {
    const alpha = Math.abs(Math.sin(Date.now() / 200));
    ctx.strokeStyle = `rgba(255, 0, 0, ${alpha})`;
    ctx.lineWidth = 4;
    ctx.strokeRect(0, 0, 640, 480);
    for (const w of state.weapons) {
      const [x, y, bw, bh] = w.box;
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, bw, bh);
      const wLabel = `WEAPON: ${w.class} (${Math.round(w.confidence * 100)}%)`;
      ctx.font = 'bold 14px Inter, sans-serif';
      const wMetrics = ctx.measureText(wLabel);
      ctx.fillStyle = 'rgba(200, 0, 0, 0.85)';
      ctx.fillRect(x, y - 24, wMetrics.width + 12, 24);
      ctx.fillStyle = '#fff';
      ctx.fillText(wLabel, x + 6, y - 6);
    }
  }

  // Tampering overlay
  if (state.tampering) {
    ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
    ctx.fillRect(0, 0, 640, 480);
    ctx.font = 'bold 28px Orbitron, sans-serif';
    ctx.fillStyle = '#ff0000';
    ctx.textAlign = 'center';
    ctx.fillText('!! CAMERA BLOCKED !!', 320, 240);
    ctx.textAlign = 'start';
  }

  // FPS info
  const infoEl = document.getElementById(`camInfo-${camId}`);
  if (infoEl) {
    infoEl.textContent = `FPS: ${state.fps.toFixed ? state.fps.toFixed(1) : state.fps} | AI Active`;
  }
});

// =============================================
//  ALERTS (existing — kept intact)
// =============================================

let alertCount = 0;

socket.on('alert', function (data) {
  // Update toast
  const alertMsg = document.getElementById('alertMessage');
  const alertToast = document.getElementById('alertToast');
  const toastTitle = document.getElementById('toastTitle');
  const toastTime = document.getElementById('toastTime');
  if (alertMsg && alertToast) {
    alertMsg.textContent = data.message;
    if (toastTitle) toastTitle.textContent = data.type === 'success' ? 'Attendance Logged' : 'Security Alert';
    if (toastTime) toastTime.textContent = new Date().toLocaleTimeString();
    const toast = new bootstrap.Toast(alertToast, { delay: 7000 });
    toast.show();
  }
  // Increment alert counter for threats only
  if (data.type === 'danger' || data.type === 'warning') {
    alertCount++;
    const el = document.getElementById('statAlertCount');
    if (el) el.textContent = alertCount;
  }
  addLogEntry(data.type || 'danger', data.message);
});

socket.on('connect', () => {
  updateCameraStatus('Connected — select a camera');
  const dot = document.getElementById('connDot');
  if (dot) { dot.style.background = '#00ff66'; dot.style.boxShadow = '0 0 8px #00ff66'; }
});

socket.on('disconnect', () => {
  updateCameraStatus('Disconnected');
  const dot = document.getElementById('connDot');
  if (dot) { dot.style.background = '#ff3b30'; dot.style.boxShadow = '0 0 8px #ff3b30'; }
});

// =============================================
//  ACTIVITY LOG
// =============================================

function addLogEntry(type, message) {
  const container = document.getElementById('attendance');
  if (!container) return;

  // Remove empty-state placeholder
  const placeholder = container.querySelector('.empty-state, .empty-log');
  if (placeholder) placeholder.remove();

  const dotClass = { danger: 'log-dot-danger', warning: 'log-dot-warning', success: 'log-dot-success', info: 'log-dot-info' }[type] || 'log-dot-info';
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.dataset.type = type;
  entry.innerHTML = `
    <span class="log-dot ${dotClass}"></span>
    <div class="log-content">
      <div class="log-msg">${message}</div>
      <div class="log-time">${now}</div>
    </div>
  `;
  container.prepend(entry);

  while (container.children.length > 60) container.removeChild(container.lastChild);

  // Apply active filter
  applyLogFilter();
}

function loadLogs() {
  fetch('/api/logs')
    .then(r => r.json())
    .then(data => {
      if (!data.logs || data.logs.length === 0) return;
      const container = document.getElementById('attendance');
      if (!container) return;
      container.innerHTML = '';
      data.logs.forEach(log => {
        addLogEntry(log.type, log.message);
      });
    })
    .catch(err => console.error('Failed to load logs:', err));
}

// =============================================
//  EMPLOYEE MANAGEMENT (existing — kept intact)
// =============================================

function loadEmployees() {
  fetch('/api/employees')
    .then(r => r.json())
    .then(data => {
      const list = document.getElementById('employeeList');
      if (!list) return;
      const count = data.employees ? data.employees.length : 0;
      const countEl = document.getElementById('statEmpCount');
      if (countEl) countEl.textContent = count;
      if (count === 0) {
        list.innerHTML = '<div class="empty-state"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg><span>No employees enrolled yet</span></div>';
        return;
      }
      list.innerHTML = data.employees.map(emp => {
        const initials = emp.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
        return `<div class="emp-card">
          <div class="emp-avatar">${initials}</div>
          <div class="emp-card-info">
            <span class="emp-card-name">${emp.name}</span>
            <span class="emp-card-id">${emp.emp_id}</span>
          </div>
        </div>`;
      }).join('');
    })
    .catch(() => {});
}

// --- Add Employee Form ---
// Log filter state
let activeFilter = 'all';
function applyLogFilter() {
  document.querySelectorAll('#attendance .log-entry').forEach(el => {
    el.style.display = (activeFilter === 'all' || el.dataset.type === activeFilter) ? '' : 'none';
  });
}

document.addEventListener('DOMContentLoaded', () => {
  enumerateCameras();

  // Start/stop camera
  const startBtn = document.getElementById('startCameraBtn');
  if (startBtn) startBtn.addEventListener('click', () => {
    startSelectedCamera();
    const badge = document.getElementById('liveBadge');
    if (badge) { badge.textContent = '● LIVE'; badge.classList.add('is-live'); }
    const statCam = document.getElementById('statCameraCount');
    if (statCam) statCam.textContent = Object.keys(activeStreams).length || 1;
  });

  const stopBtn = document.getElementById('stopCameraBtn');
  if (stopBtn) stopBtn.addEventListener('click', () => {
    stopAllStreams();
    const container = document.getElementById('videoContainer');
    if (container) container.innerHTML = '<div class="no-feed-placeholder"><div class="no-feed-icon"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg></div><p class="no-feed-title">Surveillance Stopped</p><p class="no-feed-hint">Select a camera and click <strong>Start</strong> to resume</p></div>';
    const badge = document.getElementById('liveBadge');
    if (badge) { badge.textContent = '● OFFLINE'; badge.classList.remove('is-live'); }
    const statCam = document.getElementById('statCameraCount');
    if (statCam) statCam.textContent = 0;
    updateCameraStatus('Cameras stopped');
  });

  // Log filters
  document.querySelectorAll('.log-filter').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.log-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = btn.dataset.filter;
      applyLogFilter();
    });
  });

  // Clear logs
  const clearBtn = document.getElementById('clearLogsBtn');
  if (clearBtn) clearBtn.addEventListener('click', () => {
    const container = document.getElementById('attendance');
    if (container) container.innerHTML = '<div class="empty-state mt-4"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg><span>Waiting for events...</span></div>';
  });

  loadEmployees();
  loadLogs();

  // Add Employee
  const addForm = document.getElementById('addEmployeeForm');
  if (addForm) {
    addForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const errDiv = document.getElementById('addEmployeeError');
      errDiv.classList.add('d-none');

      const fd = new FormData();
      fd.append('name', document.getElementById('addName').value);
      fd.append('emp_id', document.getElementById('addEmpId').value);
      const photoInput = document.getElementById('addPhoto');
      if (photoInput.files.length === 0) {
        errDiv.textContent = 'Please select a photo.';
        errDiv.classList.remove('d-none');
        return;
      }
      fd.append('photo', photoInput.files[0]);

      fetch('/api/employee', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
          if (data.error) {
            errDiv.textContent = data.error;
            errDiv.classList.remove('d-none');
          } else {
            bootstrap.Modal.getInstance(document.getElementById('addEmployeeModal')).hide();
            addForm.reset();
            document.getElementById('photoDropText').textContent = 'Click or drag photo here';
            loadEmployees();
            addLogEntry('success', `Employee "${data.name}" enrolled successfully.`);
          }
        })
        .catch(() => {
          errDiv.textContent = 'Network error.';
          errDiv.classList.remove('d-none');
        });
    });
  }

  // Photo preview
  const photoInput = document.getElementById('addPhoto');
  if (photoInput) {
    photoInput.addEventListener('change', function () {
      const text = document.getElementById('photoDropText');
      text.textContent = this.files.length > 0 ? this.files[0].name : 'Click or drag photo here';
    });
  }

  // Remove Employee
  const removeForm = document.getElementById('removeEmployeeForm');
  if (removeForm) {
    removeForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const errDiv = document.getElementById('removeEmployeeError');
      errDiv.classList.add('d-none');
      const empId = document.getElementById('removeEmpId').value.trim();

      fetch('/api/employee', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emp_id: empId })
      })
        .then(r => r.json())
        .then(data => {
          if (data.error) {
            errDiv.textContent = data.error;
            errDiv.classList.remove('d-none');
          } else {
            bootstrap.Modal.getInstance(document.getElementById('removeEmployeeModal')).hide();
            removeForm.reset();
            loadEmployees();
            addLogEntry('warning', `Employee "${data.name}" removed.`);
          }
        })
        .catch(() => {
          errDiv.textContent = 'Network error.';
          errDiv.classList.remove('d-none');
        });
    });
  }
});
