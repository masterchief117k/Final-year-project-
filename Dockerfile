FROM python:3.11-slim

# System deps for OpenCV (libgl1 replaces the removed libgl1-mesa-glx in Bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching — only re-runs if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run injects PORT env var; default to 8080 (GCP convention)
ENV PORT=8080
EXPOSE ${PORT}

# Single gunicorn worker with eventlet for WebSocket support
CMD exec gunicorn --worker-class eventlet -w 1 --bind "0.0.0.0:${PORT}" "project.app:socketio" --timeout 300
