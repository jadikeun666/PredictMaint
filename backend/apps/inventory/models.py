import uuid

from django.conf import settings
from django.db import models


class SparePart(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sku = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=255)
    quantity_on_hand = models.IntegerField()
    reorder_point = models.IntegerField()
    reorder_quantity = models.IntegerField()
    unit_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    supplier = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        db_table = "spare_parts"

    def __str__(self):
        return self.sku


class BearingPartCompatibility(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Free-text match against bearings.bearing_model on purpose — operators
    # can register parts before/without a formal parts catalog (database.md
    # §Consistency Rules).
    bearing_model = models.CharField(max_length=255)
    spare_part = models.ForeignKey(SparePart, on_delete=models.PROTECT, related_name="compatible_bearings")

    class Meta:
        db_table = "bearing_part_compatibility"

    def __str__(self):
        return f"{self.bearing_model} ↔ {self.spare_part.sku}"


class StockMovement(models.Model):
    class MovementType(models.TextChoices):
        RECEIPT = "RECEIPT", "Receipt"
        ISSUE = "ISSUE", "Issue"
        ADJUSTMENT = "ADJUSTMENT", "Adjustment"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    spare_part = models.ForeignKey(SparePart, on_delete=models.PROTECT, related_name="stock_movements")
    movement_type = models.CharField(max_length=10, choices=MovementType.choices)
    # Positive for RECEIPT, negative for ISSUE, signed for ADJUSTMENT (database.md).
    quantity = models.IntegerField()
    reference_work_order = models.ForeignKey(
        "scheduling.WorkOrder", on_delete=models.SET_NULL, null=True, blank=True, related_name="stock_movements"
    )
    timestamp = models.DateTimeField()
    # inventory.md §6: every ADJUSTMENT requires a non-null reason — this
    # extends the base database.md column list per that document's own note;
    # enforced at API layer for ADJUSTMENT, column itself is nullable here
    # since RECEIPT/ISSUE don't require it.
    reason = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "stock_movements"

    def __str__(self):
        return f"{self.movement_type} {self.quantity} — {self.spare_part.sku}"
