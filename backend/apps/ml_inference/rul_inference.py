"""
apps/ml_inference/rul_inference.py

RUL inference online (Sesi C, Giliran 2). Dipanggil oleh task
`apps.ml_inference.tasks.run_rul_inference` setelah klasifikasi sebuah window
berhasil. Membaca 6 titik fitur terakhir sensor itu dari InfluxDB
(`derived_features`, ml-pipeline.md §4: RUL baru keluar setelah >= 6 titik),
menjalankan model RUL terdaftar, lalu menyimpan `predictions` (Postgres) +
`predictions_ts` (InfluxDB) lengkap dengan pita ± RMSE.

KEPUTUSAN ALERT (Sesi C, Giliran 1; bukti: analyze_rul_alert_policy dan
eval_baseline_relative): RUL_ALERTING_ENABLED = False. Prediksi RUL disimpan
dan ditampilkan TETAPI tidak diteruskan ke apps.alerting. Pada evaluasi
leave-one-trajectory-out, 24 kebijakan alert (penghalusan, persistensi, tren,
gerbang FPT, fitur relatif-baseline) tidak ada yang memenuhi kriteria
(PREMATURE <= 25%, rata-rata HIT >= 50%); perilaku ambang mentah: 75% alert
CRITICAL prematur. Mengubah konstanta ini = perubahan perilaku yang harus
disertai kebijakan tervalidasi dan addendum.

PERBAIKAN (g2b, ditemukan saat verifikasi MQTT nyata): batas atas query riwayat
sebelumnya captured_at + 1 detik -> pada replay cepat ikut menarik titik MASA DEPAN
(look-ahead). Kini captured_at + 1 mikrodetik. Ditambah pemeriksaan "riwayat
mapan" (jumlah window di Postgres pada rentang jendela == jumlah titik fitur)
dengan retry Celery, supaya window yang selesai tidak berurutan (worker paralel)
tidak membentuk jendela tidak berurutan.

Torch HANYA diimpor lazy (lewat lstm_rul) di dalam worker ml_inference;
modul ini aman diimpor oleh autodiscover Celery di semua worker.
"""
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np

from apps.ml_inference import model_registry
from apps.ml_inference.rul_eval import SEQ_LEN, signed_log1p

logger = logging.getLogger(__name__)

RUL_ALERTING_ENABLED = False
HISTORY_LOOKBACK_DAYS = 30
SETTLE_MAX_RETRIES = 5
SETTLE_RETRY_COUNTDOWN_S = 2
MAX_RUL_HOURS = 1_000_000.0          # pengaman kolom numeric(10,2)

_cache = {}


def parse_time(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --- bagian murni -----------------------------------------------------------
def build_window_matrix(records, feature_order, n=SEQ_LEN):
    """records: list dict {'time', <fitur>...} urut naik. Return (X (n,F) | None, alasan | None)."""
    if len(records) < n:
        return None, "insufficient_history"
    last = records[-n:]
    rows = []
    for r in last:
        vals = [r.get(k) for k in feature_order]
        if any(v is None for v in vals):
            return None, "incomplete_features"
        rows.append([float(v) for v in vals])
    X = np.array(rows, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        return None, "incomplete_features"
    return X, None


def is_dead_window(X, feature_order) -> bool:
    """Sinyal mati: RMS (= std sinyal DC-removed) < ambang di SALAH SATU titik jendela."""
    from apps.ml_inference.ims_dataset import DEAD_SIGNAL_STD_THRESHOLD
    return bool(np.any(X[:, feature_order.index("rms")] < DEAD_SIGNAL_STD_THRESHOLD))


def scale_window(X, scaler_mean, scaler_std) -> np.ndarray:
    return ((signed_log1p(X) - np.asarray(scaler_mean)) / np.asarray(scaler_std))[None, :, :]


# --- I/O (di-patch pada test) -----------------------------------------------
def _get_model():
    if "m" not in _cache:
        entry = model_registry.get_latest_entry("RUL")
        if entry is None:
            return None
        from apps.ml_inference.lstm_rul import load_artifact
        model, tstats, mean, std, info = load_artifact(entry["artifact_path"])
        if info["meta"].get("model_version") != entry["model_version"]:
            raise RuntimeError(f"artefak {entry['artifact_path']} bukan {entry['model_version']}")
        _cache["m"] = (entry, model, tstats, mean, std, info)
        logger.info("RUL model dimuat: %s (artefak %s)", entry["model_version"], entry["artifact_path"])
    return _cache["m"]


def _predict(model, tstats, X) -> np.ndarray:
    from apps.ml_inference.lstm_rul import predict_lstm
    return predict_lstm(model, tstats, X)


def _sensor_id_for_window(window_id):
    from apps.ingestion.models import VibrationWindow
    return VibrationWindow.objects.filter(id=window_id).values_list("sensor_id", flat=True).first()


def _already_predicted(window_id, model_version) -> bool:
    from apps.ml_inference.models import Prediction
    return Prediction.objects.filter(window_id=window_id, prediction_type="RUL",
                                     model_version=model_version).exists()


def history_bounds(captured_at):
    """(start, stop) untuk query riwayat. stop EKSKLUSIF: captured_at + 1 mikrodetik
    menyertakan titik saat ini (timestamp tepat sama) dan tidak ada yang lebih baru."""
    return captured_at - timedelta(days=HISTORY_LOOKBACK_DAYS), captured_at + timedelta(microseconds=1)


def fetch_recent_features(sensor_id, captured_at, feature_order, n=SEQ_LEN):
    """n titik derived_features terakhir sensor ini pada/sebelum captured_at, urut naik."""
    from django.conf import settings
    from apps.signal_processing.influx_client import get_influx_client
    start, stop = history_bounds(captured_at)
    fields = ", ".join(f'"{f}"' for f in feature_order)
    flux = f'''
    from(bucket: "{settings.INFLUXDB_BUCKET}")
      |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
      |> filter(fn: (r) => r._measurement == "derived_features")
      |> filter(fn: (r) => r.sensor_id == "{sensor_id}")
      |> filter(fn: (r) => contains(value: r._field, set: [{fields}]))
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: {n})
    '''
    client = get_influx_client()
    try:
        records = []
        for table in client.query_api().query(flux):
            for rec in table.records:
                records.append({"time": rec.get_time(), **{k: rec.values.get(k) for k in feature_order}})
    finally:
        client.close()
    records.sort(key=lambda r: r["time"])
    return records


def _count_windows(sensor_id, start, stop) -> int:
    """Jumlah vibration_windows sensor ini dengan captured_at di [start, stop] (start=None: sejak awal)."""
    from apps.ingestion.models import VibrationWindow
    qs = VibrationWindow.objects.filter(sensor_id=sensor_id, captured_at__lte=stop)
    if start is not None:
        qs = qs.filter(captured_at__gte=start)
    return qs.count()


def _current_task():
    from celery import current_task
    try:
        return current_task._get_current_object()
    except Exception:
        return None


def _persist(bearing_id, window_id, at, pred, low, high, model_version) -> str:
    from apps.ml_inference.influx_client import write_prediction_ts
    from apps.ml_inference.models import Prediction
    q = lambda x: Decimal(str(round(float(x), 2)))
    p = Prediction.objects.create(
        window_id=window_id, bearing_id=bearing_id, prediction_type=Prediction.PredictionType.RUL,
        predicted_rul_hours=q(pred), rul_confidence_band_low_hours=q(low),
        rul_confidence_band_high_hours=q(high), model_version=model_version, predicted_at=at,
    )
    write_prediction_ts(bearing_id=bearing_id, prediction_type="RUL", model_version=model_version,
                        predicted_at=at, predicted_rul_hours=float(pred))
    return str(p.id)


# --- orkestrasi -------------------------------------------------------------
def run(bearing_id, window_id, captured_at) -> dict:
    base = {"bearing_id": str(bearing_id), "window_id": str(window_id)}
    loaded = _get_model()
    if loaded is None:
        logger.warning("run_rul_inference: tidak ada model RUL terdaftar di registry — dilewati")
        return {**base, "status": "skipped", "reason": "no_rul_model_registered"}
    entry, model, tstats, mean, std, info = loaded
    feature_order, n = list(info["feature_order"]), int(info["seq_len"])
    at = parse_time(captured_at)

    sensor_id = _sensor_id_for_window(window_id)
    if sensor_id is None:
        return {**base, "status": "skipped", "reason": "window_not_found"}
    if _already_predicted(window_id, entry["model_version"]):
        return {**base, "status": "skipped", "reason": "duplicate"}

    total = _count_windows(sensor_id, None, at)
    if total < n:                                         # benar-benar belum cukup window (ml-pipeline.md §4)
        return {**base, "status": "skipped", "reason": "insufficient_history"}
    records = fetch_recent_features(sensor_id, at, feature_order, n)
    recent = records[-n:]
    settled = len(recent) == n and _count_windows(sensor_id, recent[0]["time"], at) == n
    if not settled:                                       # ada window yang masih diproses / terlewat
        task = _current_task()
        retries = getattr(getattr(task, "request", None), "retries", None) if task is not None else None
        if task is not None and retries is not None and retries < SETTLE_MAX_RETRIES:
            raise task.retry(countdown=SETTLE_RETRY_COUNTDOWN_S, max_retries=SETTLE_MAX_RETRIES)
        logger.warning("run_rul_inference: window %s dilewati (riwayat belum mapan setelah %s percobaan)", window_id, retries)
        return {**base, "status": "skipped", "reason": "history_not_settled"}
    X, reason = build_window_matrix(recent, feature_order, n)
    if reason:
        logger.info("run_rul_inference: window %s dilewati (%s)", window_id, reason)
        return {**base, "status": "skipped", "reason": reason}
    if is_dead_window(X, feature_order):
        logger.warning("run_rul_inference: window %s dilewati (sinyal mati di jendela)", window_id)
        return {**base, "status": "skipped", "reason": "dead_signal"}

    pred = min(float(_predict(model, tstats, scale_window(X, mean, std))[0]), MAX_RUL_HOURS)
    band = float(entry["rul_confidence_band_hours"])
    low, high = max(0.0, pred - band), min(pred + band, MAX_RUL_HOURS)
    prediction_id = _persist(bearing_id, window_id, at, pred, low, high, entry["model_version"])

    # RUL_ALERTING_ENABLED = False: sengaja TIDAK enqueue apps.alerting (lihat docstring modul).
    logger.info("run_rul_inference: window %s bearing %s -> RUL %.1f jam (± %.1f, model %s, prediction_id %s, alert RUL dimatikan)",
                window_id, bearing_id, pred, band, entry["model_version"], prediction_id)
    return {**base, "status": "predicted", "predicted_rul_hours": pred, "band_low_hours": low,
            "band_high_hours": high, "model_version": entry["model_version"], "prediction_id": prediction_id}
