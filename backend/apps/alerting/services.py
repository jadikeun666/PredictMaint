"""
Alert creation + cooldown — scheduling.md §2, prd.md D1, engineering-rules.md
("Cooldown, not spam").

Keputusan implementasi (dilaporkan di akhir giliran):
- Kunci dedup = bearing_id + alert_type + severity (prd D1 menyebut
  "type/severity"; scheduling §2 menyebut bearing+type). Eskalasi HIGH ->
  CRITICAL sengaja membuat Alert baru supaya tidak tertutup alert lama.
- Window cooldown bersifat sliding: alert BELUM resolved dengan
  last_seen_at >= event_time - 24 jam dianggap masih aktif -> hanya
  last_seen_at yang diperbarui.
- Waktu event = predicted_at dari prediction (bukan wall-clock task), supaya
  hasil deterministik walau task tertunda di antrean.
- Serialisasi per bearing lewat select_for_update() pada row Bearing, supaya
  dua worker paralel tidak membuat alert ganda.
"""
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from apps.alerting.models import Alert
from apps.ml_inference.models import Prediction
from apps.registry.models import Bearing
from config import severity_thresholds as T


def parse_event_time(value) -> datetime:
    dt = value if isinstance(value, datetime) else parse_datetime(str(value))
    if dt is None:
        raise ValueError(f"predicted_at tidak bisa di-parse: {value!r}")
    if timezone.is_naive(dt):
        dt = dt.replace(tzinfo=dt_timezone.utc)
    return dt


@transaction.atomic
def upsert_alert(*, bearing_id, result, prediction_id, event_time):
    """Return (alert, created). Raises Bearing.DoesNotExist kalau bearing tidak ada."""
    Bearing.objects.select_for_update().get(pk=bearing_id)

    window_start = event_time - timedelta(hours=T.ALERT_COOLDOWN_HOURS)
    existing = (
        Alert.objects.filter(
            bearing_id=bearing_id,
            alert_type=result.alert_type,
            severity=result.severity,
            resolved_at__isnull=True,
            last_seen_at__gte=window_start,
        )
        .order_by("-last_seen_at")
        .first()
    )
    if existing is not None:
        if event_time > existing.last_seen_at:
            existing.last_seen_at = event_time
            existing.save(update_fields=["last_seen_at"])
        return existing, False

    related_id = prediction_id if Prediction.objects.filter(pk=prediction_id).exists() else None
    alert = Alert.objects.create(
        bearing_id=bearing_id,
        alert_type=result.alert_type,
        severity=result.severity,
        related_prediction_id=related_id,
        triggered_at=event_time,
        last_seen_at=event_time,
    )
    return alert, True
