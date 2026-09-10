import uuid

from django.conf import settings
from django.db import models


class ExportJob(models.Model):
    class ExportType(models.TextChoices):
        DIAGNOSTIC_PDF = "DIAGNOSTIC_PDF", "Diagnostic PDF"
        HISTORY_EXCEL = "HISTORY_EXCEL", "History Excel"

    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        RUNNING = "RUNNING", "Running"
        COMPLETED = "COMPLETED", "Completed"
        FAILED = "FAILED", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    export_type = models.CharField(max_length=20, choices=ExportType.choices)
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="export_jobs"
    )
    params = models.JSONField()
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    file_path = models.CharField(max_length=500, null=True, blank=True)
    requested_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "export_jobs"

    def __str__(self):
        return f"{self.export_type} — {self.status}"
