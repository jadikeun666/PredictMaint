"""
apps/ml_inference/influx_client.py

InfluxDB write helper untuk celery-worker-ml-inference.

Measurement yang disentuh di sini: `predictions_ts` (database.md
§InfluxDB Schema) — mirror dari row `predictions` Postgres yang relevan,
untuk charting dashboard cepat; Postgres tetap source of truth untuk
business logic (pipeline Alert/Work-Order `scheduling.md` baca dari
Postgres, bukan dari sini).

Reuse `get_influx_client()` dari apps.signal_processing.influx_client
(factory koneksi generik, tanpa logika domain apa pun) alih-alih
implementasi ulang setup koneksi InfluxDB kedua kalinya
(engineering-rules.md §Code Organization) — apps.signal_processing
sendiri TIDAK dimodifikasi, cuma di-reuse fungsi factory-nya.
"""
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from django.conf import settings

from apps.signal_processing.influx_client import get_influx_client

PREDICTIONS_TS_MEASUREMENT = "predictions_ts"


def write_prediction_ts(bearing_id, prediction_type, model_version, predicted_at,
                         fault_class=None, confidence=None, predicted_rul_hours=None):
    """
    Tulis 1 titik predictions_ts (database.md).
    Tags: bearing_id, prediction_type, model_version.
    Fields: fault_class (string, classification only), confidence (float),
    predicted_rul_hours (float, RUL only).
    """
    client = get_influx_client()
    try:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        point = (
            Point(PREDICTIONS_TS_MEASUREMENT)
            .tag("bearing_id", str(bearing_id))
            .tag("prediction_type", prediction_type)
            .tag("model_version", model_version)
            .time(predicted_at, WritePrecision.NS)
        )
        if fault_class is not None:
            point = point.field("fault_class", fault_class)
        if confidence is not None:
            point = point.field("confidence", float(confidence))
        if predicted_rul_hours is not None:
            point = point.field("predicted_rul_hours", float(predicted_rul_hours))
        write_api.write(bucket=settings.INFLUXDB_BUCKET, record=point)
    finally:
        client.close()
