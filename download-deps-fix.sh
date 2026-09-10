#!/usr/bin/env bash
set -e

cd ~/workspace/PredictMaint
source venv/bin/activate

mkdir -p ~/workspace/PredictMaint/pip-cache

pip download -d ~/workspace/PredictMaint/pip-cache \
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