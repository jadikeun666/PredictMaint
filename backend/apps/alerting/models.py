import uuid

from django.conf import settings
from django.db import models

from apps.inventory.models import SparePart
from apps.ml_inference.models import Prediction
from apps.registry.models import Bearing


class Alert(models.Model):
    class AlertType(models.TextChoices):
        FAULT_DETECTED = "FAULT_DETECTED", "Fault Detected"
        RUL_THRESHOLD = "RUL_THRESHOLD", "RUL Threshold"
        SENSOR_CONFIG_MISSING = "SENSOR_CONFIG_MISSING", "Sensor Config Missing"
        SENSOR_OFFLINE = "SENSOR_OFFLINE", "Sensor Offline"
        LOW_STOCK = "LOW_STOCK", "Low Stock"

    class Severity(models.TextChoices):
        INFO = "INFO", "Info"
        WARNING = "WARNING", "Warning"
        HIGH = "HIGH", "High"
        CRITICAL = "CRITICAL", "Critical"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bearing = models.ForeignKey(
        Bearing, on_delete=models.SET_NULL, null=True, blank=True, related_name="alerts"
    )
    spare_part = models.ForeignKey(
        SparePart, on_delete=models.SET_NULL, null=True, blank=True, related_name="alerts"
    )
    alert_type = models.CharField(max_length=25, choices=AlertType.choices)
    severity = models.CharField(max_length=10, choices=Severity.choices)
    related_prediction = models.ForeignKey(
        Prediction, on_delete=models.SET_NULL, null=True, blank=True, related_name="alerts"
    )
    triggered_at = models.DateTimeField()
    last_seen_at = models.DateTimeField()
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name="resolved_alerts"
    )

    class Meta:
        db_table = "alerts"

    def __str__(self):
        return f"{self.alert_type} ({self.severity})"
