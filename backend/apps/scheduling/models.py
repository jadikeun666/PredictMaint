import uuid

from django.conf import settings
from django.db import models

from apps.alerting.models import Alert
from apps.inventory.models import SparePart
from apps.registry.models import Bearing


class WorkOrder(models.Model):
    class Status(models.TextChoices):
        DRAFT = "DRAFT", "Draft"
        SCHEDULED = "SCHEDULED", "Scheduled"
        IN_PROGRESS = "IN_PROGRESS", "In Progress"
        COMPLETED = "COMPLETED", "Completed"
        CANCELLED = "CANCELLED", "Cancelled"

    class Priority(models.TextChoices):
        LOW = "LOW", "Low"
        MEDIUM = "MEDIUM", "Medium"
        HIGH = "HIGH", "High"
        URGENT = "URGENT", "Urgent"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bearing = models.ForeignKey(Bearing, on_delete=models.PROTECT, related_name="work_orders")
    alert = models.ForeignKey(
        Alert, on_delete=models.SET_NULL, null=True, blank=True, related_name="work_orders"
    )
    status = models.CharField(max_length=15, choices=Status.choices, default=Status.DRAFT)
    priority = models.CharField(max_length=10, choices=Priority.choices)
    scheduled_start = models.DateTimeField(null=True, blank=True)
    scheduled_end = models.DateTimeField(null=True, blank=True)
    actual_start = models.DateTimeField(null=True, blank=True)
    actual_end = models.DateTimeField(null=True, blank=True)
    assigned_technician = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name="assigned_work_orders"
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "work_orders"

    def __str__(self):
        return f"WO {self.id} — {self.bearing_id} ({self.status})"


class WorkOrderPart(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    work_order = models.ForeignKey(WorkOrder, on_delete=models.CASCADE, related_name="parts")
    spare_part = models.ForeignKey(SparePart, on_delete=models.PROTECT, related_name="work_order_usages")
    quantity_required = models.IntegerField()
    quantity_reserved = models.IntegerField(default=0)

    class Meta:
        db_table = "work_order_parts"

    def __str__(self):
        return f"{self.work_order_id} — {self.spare_part.sku} x{self.quantity_required}"
