#!/usr/bin/env bash
set -e

mkdir -p ~/workspace/PredictMaint
cd ~/workspace/PredictMaint

# ---------- Docker images ----------
docker pull postgres:16
docker pull redis:7
docker pull influxdb:2
docker pull eclipse-mosquitto:2

# ---------- Python venv + packages ----------
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip --break-system-packages

pip download --break-system-packages -d ~/workspace/PredictMaint/pip-cache \
  Django \
  djangorestframework \
  djangorestframework-simplejwt \
  channels \
  channels-redis \
  celery \
  redis \
  psycopg2-binary \
  influxdb-client \
  numpy \
  scipy \
  scikit-learn \
  torch \
  torchvision \
  paho-mqtt \
  python-dotenv \
  openpyxl \
  weasyprint \
  matplotlib \
  pandas

deactivate

# ---------- Node / frontend ----------
mkdir -p ~/workspace/PredictMaint/frontend-cache
cd ~/workspace/PredictMaint/frontend-cache

npm cache add react
npm cache add react-dom
npm cache add vite
npm cache add typescript
npm cache add recharts
npm cache add d3
npm cache add plotly.js-dist
npm cache add three
npm cache add axios

echo "SELESAI DOWNLOAD DEPENDENCIES"