"""
Celery tasks 'alerting'. Task name + kwargs process_prediction FIXED oleh
kontrak pengirim (ml-pipeline.md addendum 2026-09-20 poin 7) — jangan diubah.

Giliran 1: severity + Alert + cooldown.
Giliran 2: draft Work Order untuk Alert HIGH/CRITICAL yang BARU dibuat
(bukan untuk cooldown-update). Alert + Work Order dibuat dalam SATU transaksi:
kalau pembuatan Work Order gagal, Alert ikut di-rollback dan task bisa
diulang tanpa kehilangan Work Order (cooldown tidak akan menelannya).
"""
import logging

from celery import shared_task
from django.db import transaction

from apps.registry.models import Bearing
from apps.scheduling.services import create_draft_work_order

from .services import parse_event_time, upsert_alert
from .severity import compute_severity

logger = logging.getLogger(__name__)


@shared_task(name="apps.alerting.tasks.process_prediction")
def process_prediction(
    prediction_id,
    bearing_id,
    prediction_type,
    fault_class,
    classification_confidence,
    predicted_rul_hours,
    model_version,
    predicted_at,
):
    if prediction_type == "CLASSIFICATION":
        result = compute_severity(fault_class=fault_class, confidence=classification_confidence)
    elif prediction_type == "RUL":
        result = compute_severity(rul_hours=predicted_rul_hours)
    else:
        logger.warning("[ALERTING] prediction_type tidak dikenal %r prediction_id=%s", prediction_type, prediction_id)
        return {"status": "skipped", "reason": "unknown_prediction_type"}

    if result is None:
        return {"status": "no_alert", "prediction_id": prediction_id}

    try:
        with transaction.atomic():
            alert, created = upsert_alert(
                bearing_id=bearing_id,
                result=result,
                prediction_id=prediction_id,
                event_time=parse_event_time(predicted_at),
            )
            work_order = create_draft_work_order(alert) if created else None
    except Bearing.DoesNotExist:
        logger.error("[ALERTING] bearing_id=%s tidak ditemukan, prediction_id=%s di-skip", bearing_id, prediction_id)
        return {"status": "skipped", "reason": "bearing_not_found"}

    logger.info(
        "[ALERTING] %s alert_id=%s type=%s severity=%s bearing_id=%s model=%s work_order_id=%s",
        "CREATED" if created else "cooldown-update",
        alert.id, result.alert_type, result.severity, bearing_id, model_version,
        work_order.id if work_order else None,
    )
    return {
        "status": "created" if created else "cooldown",
        "alert_id": str(alert.id),
        "severity": str(result.severity),
        "alert_type": str(result.alert_type),
        "work_order_id": str(work_order.id) if work_order else None,
    }
