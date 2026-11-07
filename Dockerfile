FROM python:3.11-slim-bookworm

# Install FFmpeg with libass support, fonts, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libass9 \
    libass-dev \
    fonts-liberation \
    fonts-dejavu-core \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY vtpl ./vtpl

# Install project and web dependencies
RUN pip install --no-cache-dir -e '.[dev]' \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    jinja2

# Pre-download the default Whisper model so first user analysis is fast and memory-efficient
ENV VTPL_WHISPER_MODEL=tiny
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"

# Create work directory
RUN mkdir -p /app/work

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Start FastAPI production server
CMD uvicorn vtpl.web.app:app --host ${HOST} --port ${PORT}
