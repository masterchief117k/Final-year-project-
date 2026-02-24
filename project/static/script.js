var socket = io.connect('http://localhost:5000');

socket.on('alert', function(data) {
    document.getElementById('alertMessage').innerText = data.message;
    var toast = new bootstrap.Toast(document.getElementById('alertToast'));
    toast.show();
});
