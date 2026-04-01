var socket = io.connect();

// --- On Page Load ---
document.addEventListener('DOMContentLoaded', function () {
    // Load historical logs
    fetch('/api/logs')
        .then(response => response.json())
        .then(data => {
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(log => { addLogEntry(log, false); });
            } else {
                var box = document.getElementById('attendance');
                if (box && !box.querySelector('.log-entry:not(.empty-log)')) {
                    // leave "Waiting for events..." placeholder
                }
            }
        })
        .catch(err => console.error("Error fetching logs:", err));

    // Load employee list
    loadEmployees();

    // Photo drop zone — show filename when selected
    var photoInput = document.getElementById('addPhoto');
    if (photoInput) {
        photoInput.addEventListener('change', function () {
            var text = document.getElementById('photoDropText');
            if (photoInput.files.length > 0) {
                text.textContent = '✓ ' + photoInput.files[0].name;
                text.style.color = '#00f0ff';
            } else {
                text.textContent = 'Click or drag photo here';
                text.style.color = '#8e9bb0';
            }
        });
    }

    // Add Employee form
    var addForm = document.getElementById('addEmployeeForm');
    if (addForm) {
        addForm.addEventListener('submit', function (e) {
            e.preventDefault();
            var name = document.getElementById('addName').value.trim();
            var empId = document.getElementById('addEmpId').value.trim();
            var photo = document.getElementById('addPhoto').files[0];
            var errDiv = document.getElementById('addEmployeeError');
            errDiv.classList.add('d-none');

            if (!name || !empId || !photo) {
                errDiv.textContent = 'All fields (name, ID, photo) are required.';
                errDiv.classList.remove('d-none');
                return;
            }

            var formData = new FormData();
            formData.append('name', name);
            formData.append('emp_id', empId);
            formData.append('photo', photo);

            var btn = addForm.querySelector('button[type=submit]');
            btn.disabled = true;
            btn.textContent = 'Registering...';

            fetch('/api/employee', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    btn.disabled = false;
                    btn.textContent = 'Register Employee';
                    if (data.error) {
                        errDiv.textContent = data.error;
                        errDiv.classList.remove('d-none');
                    } else {
                        // Close modal and reset
                        bootstrap.Modal.getInstance(document.getElementById('addEmployeeModal')).hide();
                        addForm.reset();
                        document.getElementById('photoDropText').textContent = 'Click or drag photo here';
                        document.getElementById('photoDropText').style.color = '#8e9bb0';
                        loadEmployees();
                    }
                })
                .catch(() => {
                    btn.disabled = false;
                    btn.textContent = 'Register Employee';
                    errDiv.textContent = 'Network error. Please try again.';
                    errDiv.classList.remove('d-none');
                });
        });
    }

    // Remove Employee form
    var removeForm = document.getElementById('removeEmployeeForm');
    if (removeForm) {
        removeForm.addEventListener('submit', function (e) {
            e.preventDefault();
            var empId = document.getElementById('removeEmpId').value.trim();
            var errDiv = document.getElementById('removeEmployeeError');
            errDiv.classList.add('d-none');

            if (!empId) {
                errDiv.textContent = 'Employee ID is required.';
                errDiv.classList.remove('d-none');
                return;
            }

            var btn = removeForm.querySelector('button[type=submit]');
            btn.disabled = true;
            btn.textContent = 'Removing...';

            fetch('/api/employee', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ emp_id: empId })
            })
                .then(r => r.json())
                .then(data => {
                    btn.disabled = false;
                    btn.textContent = 'Remove Employee';
                    if (data.error) {
                        errDiv.textContent = data.error;
                        errDiv.classList.remove('d-none');
                    } else {
                        bootstrap.Modal.getInstance(document.getElementById('removeEmployeeModal')).hide();
                        removeForm.reset();
                        loadEmployees();
                    }
                })
                .catch(() => {
                    btn.disabled = false;
                    btn.textContent = 'Remove Employee';
                    errDiv.textContent = 'Network error. Please try again.';
                    errDiv.classList.remove('d-none');
                });
        });
    }
});

function loadEmployees() {
    fetch('/api/employees')
        .then(r => r.json())
        .then(data => { renderEmployees(data.employees || []); })
        .catch(err => console.error('Error loading employees:', err));
}

function renderEmployees(employees) {
    var list = document.getElementById('employeeList');
    if (!list) return;
    list.innerHTML = '';
    if (employees.length === 0) {
        list.innerHTML = '<div class="empty-log" style="margin-top:16px;">No employees registered yet.</div>';
        return;
    }
    employees.forEach(function (emp) {
        var initials = emp.name.split(' ').map(function (p) { return p[0]; }).join('').toUpperCase().slice(0, 2);
        var card = document.createElement('div');
        card.className = 'emp-card';
        card.innerHTML =
            '<div class="emp-avatar">' + initials + '</div>' +
            '<div class="emp-card-info">' +
            '<div class="emp-name">' + emp.name + '</div>' +
            '<div class="emp-id-tag">' + emp.emp_id + '</div>' +
            '</div>';
        list.appendChild(card);
    });
}



function addLogEntry(data, prepend) {
    var attendanceBox = document.getElementById('attendance');
    if (!attendanceBox) return;

    var emptyLog = attendanceBox.querySelector('.empty-log');
    if (emptyLog) emptyLog.remove();

    var logEntry = document.createElement('div');
    logEntry.className = 'log-entry';

    var timeStr;
    if (data.timestamp) {
        // SQLite timestamp is UTC
        var d = new Date(data.timestamp + 'Z');
        timeStr = d.toLocaleTimeString();
    } else {
        var now = new Date();
        timeStr = now.toLocaleTimeString();
    }

    var alertType = data.type || 'danger';
    var tagColor = 'red';
    var tagLabel = 'ALERT';
    if (alertType === 'warning') {
        tagColor = '#ffaa00';
        tagLabel = 'WARNING';
    } else if (alertType === 'success') {
        tagColor = '#00ff66';
        tagLabel = 'LOG';
    }

    logEntry.innerHTML = '<span style="color:' + tagColor + ';font-weight:bold">[' + tagLabel + ']</span> '
        + '<span style="color:#555;font-size:12px">' + timeStr + '</span> - '
        + data.message;

    if (prepend) {
        attendanceBox.prepend(logEntry);
        logEntry.style.transition = 'background 0.5s';
        logEntry.style.background = 'rgba(0, 240, 255, 0.1)';
        setTimeout(function () {
            logEntry.style.background = 'transparent';
        }, 1000);
    } else {
        attendanceBox.appendChild(logEntry);
    }
}

socket.on('alert', function (data) {
    var alertType = data.type || 'danger';

    // --- Update toast style based on alert type ---
    var toastEl = document.getElementById('alertToast');
    var toastHeader = toastEl.querySelector('.toast-header');
    var alertTitle = toastEl.querySelector('.toast-header strong');

    // Reset classes
    toastHeader.className = 'toast-header bg-dark text-white';

    if (alertType === 'danger') {
        toastHeader.classList.add('border-bottom', 'border-danger');
        alertTitle.className = 'me-auto text-danger';
        alertTitle.innerText = 'Security Alert';
    } else if (alertType === 'warning') {
        toastHeader.classList.add('border-bottom', 'border-warning');
        alertTitle.className = 'me-auto text-warning';
        alertTitle.innerText = 'Camera Warning';
    } else if (alertType === 'success') {
        toastHeader.classList.add('border-bottom', 'border-success');
        alertTitle.className = 'me-auto text-success';
        alertTitle.innerText = 'Attendance';
    }

    // Show toast
    document.getElementById('alertMessage').innerText = data.message;
    var toast = new bootstrap.Toast(document.getElementById('alertToast'));
    toast.show();

    // --- Append to Activity Logs ---
    addLogEntry(data, true);

    // --- Play audio beep for danger alerts ---
    if (alertType === 'danger') {
        try {
            var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            var oscillator = audioCtx.createOscillator();
            var gainNode = audioCtx.createGain();
            oscillator.connect(gainNode);
            gainNode.connect(audioCtx.destination);
            oscillator.type = 'sine';
            oscillator.frequency.value = 880;
            gainNode.gain.value = 0.3;
            oscillator.start();
            setTimeout(function () { oscillator.stop(); }, 200);
        } catch (e) { }
    }
});
