import uuid

from django.db import models

from apps.registry.models import Sensor


class VibrationWindow(models.Model):
    """
    Immutable — insert-only (engineering-rules.md §Data Integrity,
    database.md). No UPDATE statements target this table in application
    code; corrections are handled by inserting a new row. Not enforced at
    the DB layer here (no trigger) — enforced by never exposing a PATCH/PUT
    endpoint for this resource (api.md §Write Restrictions), consistent
    with the "thin ingestion, fat worker" and no-manual-write rules.
    """

    class SourceDataset(models.TextChoices):
        CWRU = "CWRU", "CWRU"
        PU = "PU", "Paderborn"
        IMS = "IMS", "IMS"
        SIMULATED_SYNTHETIC = "SIMULATED_SYNTHETIC", "Simulated Synthetic"
        HARDWARE = "HARDWARE", "Hardware"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sensor = models.ForeignKey(Sensor, on_delete=models.PROTECT, related_name="vibration_windows")
    captured_at = models.DateTimeField()
    duration_s = models.DecimalField(max_digits=6, decimal_places=3)
    num_samples = models.IntegerField()
    waveform_path = models.CharField(max_length=500)
    source_dataset = models.CharField(max_length=25, choices=SourceDataset.choices)
    ingested_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vibration_windows"

    def __str__(self):
        return f"{self.sensor_id} @ {self.captured_at}"
