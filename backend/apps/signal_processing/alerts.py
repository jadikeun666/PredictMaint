"""
SENSOR_CONFIG_MISSING alert cooldown + auto-resolve
(signal-processing.md §Pipeline step 7, prd.md A2, engineering-rules.md
"Cooldown, not spam"). Fresh implementation against
apps.alerting.models.Alert — NOT importing apps.ingestion (BATASAN 1).
Severity=WARNING is an implementation decision (see chat for rationale).
"""
from django.utils import timezone
from apps.alerting.models import Alert


def upsert_sensor_config_missing_alert(bearing):
    now = timezone.now()
    existing = Alert.objects.filter(
        bearing=bearing,
        alert_type=Alert.AlertType.SENSOR_CONFIG_MISSING,
        resolved_at__isnull=True,
    ).first()
    if existing:
        existing.last_seen_at = now
        existing.save(update_fields=["last_seen_at"])
        return existing, False

    alert = Alert.objects.create(
        bearing=bearing,
        alert_type=Alert.AlertType.SENSOR_CONFIG_MISSING,
        severity=Alert.Severity.WARNING,
        triggered_at=now,
        last_seen_at=now,
    )
    return alert, True


def resolve_sensor_config_missing_alert(bearing):
    now = timezone.now()
    Alert.objects.filter(
        bearing=bearing,
        alert_type=Alert.AlertType.SENSOR_CONFIG_MISSING,
        resolved_at__isnull=True,
    ).update(resolved_at=now, resolved_by=None)
