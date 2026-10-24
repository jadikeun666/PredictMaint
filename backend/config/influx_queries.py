"""
Pembacaan InfluxDB bersama (api.md: GET /bearings/{id}/derived-features/ dan
/predictions-trend/). Dipakai ulang oleh worker export (grafik RUL PDF) supaya angka di
dashboard dan PDF dijamin dari query yang sama (api.md).

Hasil: (points, truncated). Titik terbaru dipertahankan (query desc + limit+1), lalu
dibalik menjadi urutan menaik untuk charting. truncated=True berarti titik LEBIH LAMA
dalam rentang tidak ikut terkirim -> persempit rentang.
Field null (tidak ditulis) dikembalikan sebagai None.
"""
import logging
import uuid
from datetime import timezone as dt_timezone

from django.conf import settings

from apps.ml_inference.influx_client import PREDICTIONS_TS_MEASUREMENT
from apps.signal_processing.influx_client import (
    DERIVED_FEATURES_FIELD_NAMES,
    DERIVED_FEATURES_MEASUREMENT,
    get_influx_client,
)
from config.api_errors import ServiceUnavailable

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 5000
MAX_LIMIT = 20000
PREDICTION_TYPES = ("CLASSIFICATION", "RUL")
PREDICTION_FIELD_NAMES = ("fault_class", "confidence", "predicted_rul_hours")


def _flux_time(dt):
    return dt.astimezone(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_flux(measurement, bearing_id, start, end, limit, row_key, prediction_type=None):
    """Semua nilai yang masuk string Flux sudah divalidasi (UUID, enum, int, datetime)."""
    bearing_id = str(uuid.UUID(str(bearing_id)))
    extra = ""
    if prediction_type is not None:
        if prediction_type not in PREDICTION_TYPES:
            raise ValueError(f"prediction_type tidak valid: {prediction_type!r}")
        extra = f' and r.prediction_type == "{prediction_type}"'
    return (
        f'from(bucket: "{settings.INFLUXDB_BUCKET}")\n'
        f"  |> range(start: {_flux_time(start)}, stop: {_flux_time(end)})\n"
        f'  |> filter(fn: (r) => r._measurement == "{measurement}" and r.bearing_id == "{bearing_id}"{extra})\n'
        f'  |> pivot(rowKey: {row_key}, columnKey: ["_field"], valueColumn: "_value")\n'
        f"  |> group()\n"
        f'  |> sort(columns: ["_time"], desc: true)\n'
        f"  |> limit(n: {int(limit) + 1})\n"
    )


def _run(flux):
    client = None
    try:
        client = get_influx_client()
        tables = client.query_api().query(flux)
        return [rec for table in tables for rec in table.records]
    except Exception as exc:  # koneksi/otentikasi/query gagal -> 503, detail hanya di log
        logger.exception("InfluxDB query gagal")
        raise ServiceUnavailable("InfluxDB tidak dapat dijangkau atau query gagal.") from exc
    finally:
        if client is not None:
            client.close()


def _finish(rows, limit, shape):
    truncated = len(rows) > limit
    kept = rows[:limit]  # rows terurut menurun -> yang terbaru dipertahankan
    kept.reverse()       # keluaran menaik
    return [shape(r) for r in kept], truncated


def query_derived_features(*, bearing_id, start, end, limit=DEFAULT_LIMIT):
    flux = build_flux(DERIVED_FEATURES_MEASUREMENT, bearing_id, start, end, limit, '["_time"]')

    def shape(rec):
        point = {"captured_at": rec.get_time()}
        for name in DERIVED_FEATURES_FIELD_NAMES:
            point[name] = rec.values.get(name)
        return point

    return _finish(_run(flux), limit, shape)


def query_predictions_trend(*, bearing_id, start, end, limit=DEFAULT_LIMIT, prediction_type=None):
    flux = build_flux(
        PREDICTIONS_TS_MEASUREMENT, bearing_id, start, end, limit,
        '["_time", "model_version", "prediction_type"]', prediction_type,
    )

    def shape(rec):
        point = {
            "predicted_at": rec.get_time(),
            "prediction_type": rec.values.get("prediction_type"),
            "model_version": rec.values.get("model_version"),
        }
        for name in PREDICTION_FIELD_NAMES:
            point[name] = rec.values.get(name)
        return point

    return _finish(_run(flux), limit, shape)
