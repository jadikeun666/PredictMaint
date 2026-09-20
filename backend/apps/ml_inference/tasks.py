"""
apps/ml_inference/tasks.py

Celery task `run_inference` — entry point untuk queue `ml_inference`.
Kontrak task name + kwargs FIXED dari sisi pengirim
(apps.signal_processing.tasks, sesi lalu, sudah terverifikasi end-to-end
via MQTT nyata — claude.md):

    apps.ml_inference.tasks.run_inference(
        bearing_id=<uuid str>, window_id=<uuid str>,
        captured_at=<isoformat str>,
        features=<dict 9 nilai kanonik, bpfo/bpfi/bsf/ftf_amp BISA None>,
    )

TIDAK boleh mengubah task name atau kwargs signature ini.

Giliran 4 (sesi ini): setelah klasifikasi, tulis row `predictions`
(Postgres) + `predictions_ts` (InfluxDB), lalu enqueue
apps.alerting.tasks.process_prediction — task itu BELUM ada (pola sama
seperti run_inference sebelum sesi ini: sender enqueue dulu via nama
task, task-nya dibangun di sesi berikutnya). Kontrak kwargs yang dikirim
ke alerting didokumentasikan di docstring enqueue di bawah — JANGAN
diubah tanpa mengoordinasikan ulang dengan sesi yang akan membangun
apps.alerting.tasks.
"""
import logging

import joblib
from celery import current_app, shared_task

from apps.ml_inference import model_registry
from apps.ml_inference.feature_extraction import engineer_features
from apps.ml_inference.influx_client import write_prediction_ts
from apps.ml_inference.persistence import save_classification_prediction

logger = logging.getLogger(__name__)

# ml-pipeline.md §1 poin 5: confidence < threshold -> fault_class = UNKNOWN,
# diterapkan di INFERENCE time, bukan saat training/eval scoring (training
# script tidak menerapkan threshold ini sama sekali, sesuai spec).
CONFIDENCE_THRESHOLD = 0.6

# Fitur fault-frequency yang BISA null dari signal_processing (kondisi
# SENSOR_CONFIG_MISSING — geometri bearing belum lengkap ATAU belum ada
# heartbeat RPM sama sekali, lihat database.md addendum 2026-09-10).
FAULT_FREQ_KEYS = ("bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp")

# Cache per PROSES worker (bukan per-task) — lihat _get_classifier().
_classifier_cache = {}


def _get_classifier():
    """
    Load model rf-baseline SEKALI per worker process
    (architecture.md §Model Loading), LAZY: dimuat saat task PERTAMA
    benar-benar dieksekusi di worker ini, bukan saat import module.

    Keputusan implementasi (dilaporkan, bukan blocking): tasks.py ini
    ikut ter-import oleh autodiscover_tasks() Celery di SEMUA worker
    (signal_processing/alerting/export/ml_inference — pelajaran
    architecture.md addendum 2026-09-10). Kalau model di-load di level
    module (saat import), SEMUA worker ikut memuat model ke memory
    walau cuma ml_inference yang butuh. Dengan pola lazy-cache ini,
    _get_classifier() hanya benar-benar dipanggil dari dalam
    run_inference() — dan run_inference() cuma pernah dieksekusi oleh
    worker yang memang men-subscribe queue ml_inference. Worker lain
    tetap aman meng-import module ini (joblib/scikit-learn sudah
    terverifikasi ada di image semua worker) tapi tidak pernah memuat
    model apa pun ke memory-nya.
    """
    if "clf" not in _classifier_cache:
        entry = model_registry.get_latest_entry("CLASSIFICATION")
        if entry is None:
            raise RuntimeError(
                "Tidak ada model CLASSIFICATION terdaftar di Model Registry "
                f"({model_registry.REGISTRY_PATH}) — jalankan training dulu: "
                "python manage.py train_rf_baseline"
            )
        clf = joblib.load(entry["artifact_path"])
        _classifier_cache["clf"] = clf
        _classifier_cache["entry"] = entry
        logger.info(
            "rf-baseline model dimuat: %s (eval_accuracy=%.4f, eval_macro_f1=%.4f, artifact=%s)",
            entry["model_version"], entry["eval_accuracy"], entry["eval_macro_f1"],
            entry["artifact_path"],
        )
    return _classifier_cache["clf"], _classifier_cache["entry"]


def _build_feature_vector(features: dict, feature_order: list) -> list:
    """
    Bangun input vector sesuai `feature_order` PERSIS yang tercatat di
    entry Model Registry model yang dipakai (dibaca dari registry, BUKAN
    hardcode ulang di sini) — supaya kalau MODEL_FEATURE_ORDER berubah
    lagi di training sesi depan, run_inference otomatis ikut sinkron
    tanpa perlu edit manual di file ini.
    """
    engineered = engineer_features(features)
    full_feats = {**features, **engineered}
    return [full_feats[k] for k in feature_order]


def _enqueue_alerting(prediction, bearing_id, model_version, predicted_at):
    """
    Enqueue apps.alerting.tasks.process_prediction — task BELUM
    diimplementasikan sesi ini (lihat docstring modul). Dipanggil via
    current_app.send_task() dengan STRING nama task (bukan import
    langsung objek task), karena modul apps.alerting.tasks belum ada —
    pola yang sama persis dipakai apps.signal_processing sebelum
    apps.ml_inference.tasks.run_inference ini sendiri dibangun.

    KONTRAK kwargs untuk apps.alerting.tasks.process_prediction (harus
    dipakai persis ini oleh sesi yang membangun task-nya nanti):

        apps.alerting.tasks.process_prediction(
            prediction_id=<uuid str>,
            bearing_id=<uuid str>,
            prediction_type="CLASSIFICATION",
            fault_class=<str>,
            classification_confidence=<float>,
            predicted_rul_hours=None,
            model_version=<str>,
            predicted_at=<isoformat str>,
        )

    Routing ke queue "alerting" sudah otomatis lewat task_routes di
    config/celery.py (pola "apps.alerting.tasks.*"), tidak perlu
    parameter queue= eksplisit di sini.
    """
    current_app.send_task(
        "apps.alerting.tasks.process_prediction",
        kwargs={
            "prediction_id": str(prediction.id),
            "bearing_id": str(bearing_id),
            "prediction_type": "CLASSIFICATION",
            "fault_class": prediction.fault_class,
            "classification_confidence": float(prediction.classification_confidence),
            "predicted_rul_hours": None,
            "model_version": model_version,
            "predicted_at": predicted_at,
        },
    )


@shared_task(name="apps.ml_inference.tasks.run_inference")
def run_inference(bearing_id: str, window_id: str, captured_at: str, features: dict):
    """
    Klasifikasi 1 window (ml-pipeline.md §1). Lihat FAULT_FREQ_KEYS di
    atas untuk kondisi skip.
    """
    missing = [k for k in FAULT_FREQ_KEYS if features.get(k) is None]
    if missing:
        # Keputusan (claude.md addendum 2026-09-11, missing-feature-at-
        # inference): SKIP klasifikasi untuk window ini, JANGAN fabrikasi
        # nilai pengganti (engineering-rules.md). Tidak ada row
        # `predictions` ditulis — analog pola "insufficient history" RUL
        # (ml-pipeline.md §4): sistem tetap jalan, window ini saja yang
        # dilewati.
        logger.warning(
            "run_inference: skip window %s (bearing %s) — fitur fault-frequency null: %s",
            window_id, bearing_id, missing,
        )
        return {
            "status": "skipped",
            "reason": "missing_fault_frequency_features",
            "missing_features": missing,
            "bearing_id": bearing_id,
            "window_id": window_id,
        }

    clf, entry = _get_classifier()
    feature_order = entry["feature_order"]
    vector = _build_feature_vector(features, feature_order)

    probs = clf.predict_proba([vector])[0]
    classes = clf.classes_
    best_idx = probs.argmax()
    confidence = float(probs[best_idx])
    raw_predicted_class = str(classes[best_idx])

    # confidence < threshold -> UNKNOWN (ml-pipeline.md §1 poin 5). UNKNOWN
    # TETAP disimpan sebagai row prediction sungguhan (bukan di-skip) —
    # scheduling.md §1 sendiri yang memutuskan "UNKNOWN -> no alert", jadi
    # logika severity tetap satu tempat (apps.alerting), tidak diduplikasi
    # ke sini.
    if confidence < CONFIDENCE_THRESHOLD:
        fault_class = "UNKNOWN"
    else:
        fault_class = raw_predicted_class

    prediction = save_classification_prediction(
        bearing_id=bearing_id,
        window_id=window_id,
        captured_at=captured_at,
        fault_class=fault_class,
        confidence=confidence,
        model_version=entry["model_version"],
    )

    write_prediction_ts(
        bearing_id=bearing_id,
        prediction_type="CLASSIFICATION",
        model_version=entry["model_version"],
        predicted_at=prediction.predicted_at,
        fault_class=fault_class,
        confidence=confidence,
    )

    _enqueue_alerting(
        prediction=prediction,
        bearing_id=bearing_id,
        model_version=entry["model_version"],
        predicted_at=captured_at,
    )

    logger.info(
        "run_inference: window %s bearing %s -> %s (confidence=%.3f, raw=%s, model=%s, prediction_id=%s)",
        window_id, bearing_id, fault_class, confidence, raw_predicted_class,
        entry["model_version"], prediction.id,
    )

    return {
        "status": "classified",
        "bearing_id": bearing_id,
        "window_id": window_id,
        "captured_at": captured_at,
        "fault_class": fault_class,
        "confidence": confidence,
        "raw_predicted_class": raw_predicted_class,
        "model_version": entry["model_version"],
        "prediction_id": str(prediction.id),
    }
