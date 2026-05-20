FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 nginx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Nginx config for SSL termination + reverse proxy
COPY nginx.conf /etc/nginx/sites-available/default

EXPOSE 80 443

# Start nginx + gunicorn
CMD nginx && gunicorn --worker-class eventlet -w 1 --bind 127.0.0.1:5000 "project.app:socketio" --timeout 120
