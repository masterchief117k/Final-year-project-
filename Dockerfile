FROM python:3.11-slim

# System deps for OpenCV (libgl1 replaces the removed libgl1-mesa-glx in Bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT env var; default to 5000 for local dev
ENV PORT=5000
EXPOSE ${PORT}

# Render handles SSL & reverse proxy — just run gunicorn directly
CMD gunicorn --worker-class eventlet -w 1 --bind "0.0.0.0:${PORT}" "project.app:socketio" --timeout 120
