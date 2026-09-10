FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf2.0-0 libffi8 shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
COPY pip-cache /tmp/pip-cache

RUN pip install --no-cache-dir --no-index --find-links=/tmp/pip-cache -r requirements.txt \
    && rm -rf /tmp/pip-cache

COPY backend/ .

EXPOSE 8000
