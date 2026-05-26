# services/ingestion-worker/Dockerfile
# Multi-stage image for the Python ingestion worker. Slim base + native libs
# required by python-magic, OpenCV, and `unstructured` (poppler, tesseract).

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        ffmpeg \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app

# Default: run as a long-lived worker process. Overridden by docker-compose
# in prod to a `celery -A worker worker` invocation.
CMD ["python", "-m", "pipeline"]
