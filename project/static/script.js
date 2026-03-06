var socket = io.connect('http://localhost:5000');

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
    var attendanceBox = document.getElementById('attendance');
    if (attendanceBox) {
        // Remove empty placeholder
        var emptyLog = attendanceBox.querySelector('.empty-log');
        if (emptyLog) emptyLog.remove();

        var logEntry = document.createElement('div');
        logEntry.className = 'log-entry';

        var now = new Date();
        var timeStr = now.toLocaleTimeString();

        // Color-code log entries
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

        attendanceBox.prepend(logEntry);

        // Flash effect on new entry
        logEntry.style.transition = 'background 0.5s';
        logEntry.style.background = 'rgba(0, 240, 255, 0.1)';
        setTimeout(function () {
            logEntry.style.background = 'transparent';
        }, 1000);
    }

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
