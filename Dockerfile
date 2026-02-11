FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="Personal AI Platform"
LABEL description="Whisper Unified - STT + Speaker Diarization + Cache in one container"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (use built-in Python 3.10)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Copy requirements first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY orchestrator.py .

RUN mkdir -p /tmp/audio && chmod 777 /tmp/audio
RUN mkdir -p /root/.cache/huggingface && chmod 777 /root/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "src.whisper_unified"]
