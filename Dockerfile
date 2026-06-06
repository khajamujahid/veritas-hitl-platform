# ============================================================================
# VERITAS HITL Platform - Production Dockerfile
# Multi-stage build: Frontend (Node) → Backend (Python)
# ============================================================================

# ─── STAGE 1: Build React Frontend ───────────────────────────────────────────
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (better caching)
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --production=false

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# ─── STAGE 2: Python Backend + Serve Frontend ────────────────────────────────
FROM python:3.11-slim

# Install system dependencies for OpenCV and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY server.py .
COPY app.py .
COPY image_forensics.py .
COPY video_forensics.py .
COPY tasks.py .
COPY server_with_celery.py .

# Copy built frontend from Stage 1
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Create required directories
RUN mkdir -p uploads forensic_output analysis_results forensic_output/extracted_frames

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run with gunicorn
CMD ["gunicorn", "server_with_celery:app", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "2", "--threads", "4"]
