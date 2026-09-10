import uuid

from django.db import models

from apps.ingestion.models import VibrationWindow
from apps.registry.models import Bearing


class Prediction(models.Model):
    class PredictionType(models.TextChoices):
        CLASSIFICATION = "CLASSIFICATION", "Classification"
        RUL = "RUL", "Remaining Useful Life"

    class FaultClass(models.TextChoices):
        HEALTHY = "HEALTHY", "Healthy"
        INNER_RACE = "INNER_RACE", "Inner Race"
        OUTER_RACE = "OUTER_RACE", "Outer Race"
        BALL = "BALL", "Ball"
        UNKNOWN = "UNKNOWN", "Unknown"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Nullable — RUL predictions may aggregate multiple windows (database.md).
    window = models.ForeignKey(
        VibrationWindow, on_delete=models.SET_NULL, null=True, blank=True, related_name="predictions"
    )
    bearing = models.ForeignKey(Bearing, on_delete=models.PROTECT, related_name="predictions")
    prediction_type = models.CharField(max_length=20, choices=PredictionType.choices)
    fault_class = models.CharField(max_length=20, choices=FaultClass.choices, null=True, blank=True)
    classification_confidence = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    predicted_rul_hours = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    rul_confidence_band_low_hours = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    rul_confidence_band_high_hours = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    # Always populated — inference is never served from an unversioned model
    # (ml-pipeline.md §3 Model Registry Contract, engineering-rules.md).
    model_version = models.CharField(max_length=100)
    predicted_at = models.DateTimeField()

    class Meta:
        db_table = "predictions"

    def __str__(self):
        return f"{self.prediction_type} — {self.bearing_id} @ {self.predicted_at}"
