"""
apps/ml_inference/persistence.py

Postgres write helper untuk `Prediction` (database.md `predictions`
table). Dipisah dari tasks.py supaya orkestrasi Celery (tasks.py) tetap
tipis dan bagian ini independen-testable tanpa perlu konteks Celery task
yang terbind.
"""
from decimal import Decimal

from django.utils.dateparse import parse_datetime

from apps.ml_inference.models import Prediction


def save_classification_prediction(*, bearing_id, window_id, captured_at,
                                     fault_class, confidence, model_version):
    """
    Simpan 1 row prediction CLASSIFICATION (database.md `predictions`).

    `predicted_at` = captured_at dari window sumber (konsisten dengan pola
    signal_processing men-timestamp derived_features — supaya chart/
    export selaras dengan kapan vibrasi ASLI terjadi, bukan kapan worker
    kebetulan sempat memprosesnya).

    `window_id` boleh None (field nullable — database.md: "RUL predictions
    may aggregate multiple windows"; di sini dipakai juga untuk kasus
    window sintetis/test yang tidak punya row `vibration_windows`
    sungguhan, supaya tidak memaksakan FK ke row yang tidak ada).
    """
    if isinstance(captured_at, str):
        predicted_at = parse_datetime(captured_at)
    else:
        predicted_at = captured_at

    return Prediction.objects.create(
        window_id=window_id,
        bearing_id=bearing_id,
        prediction_type=Prediction.PredictionType.CLASSIFICATION,
        fault_class=fault_class,
        classification_confidence=Decimal(str(round(confidence, 3))),
        model_version=model_version,
        predicted_at=predicted_at,
    )
