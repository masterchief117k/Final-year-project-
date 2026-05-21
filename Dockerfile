FROM python:3.11-slim

# System deps for OpenCV (libgl1 replaces deprecated libgl1-mesa-glx in Bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create persistent data directory (Render Disk mounts here)
RUN mkdir -p /data/employees

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render injects PORT env var (default 10000)
ENV PORT=10000
ENV DATA_DIR=/data
EXPOSE ${PORT}

# Gunicorn with eventlet worker for WebSocket support
# - 1 worker: memory-conscious for AI models
# - 300s timeout: keeps WebSocket connections alive
CMD exec gunicorn --worker-class eventlet -w 1 --bind "0.0.0.0:${PORT}" "project.app:socketio" --timeout 300
