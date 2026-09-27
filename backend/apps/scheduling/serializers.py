from rest_framework import serializers

from .models import WorkOrder, WorkOrderPart


class WorkOrderPartSerializer(serializers.ModelSerializer):
    spare_part_sku = serializers.CharField(source="spare_part.sku", read_only=True)

    class Meta:
        model = WorkOrderPart
        fields = [
            "id",
            "spare_part",
            "spare_part_sku",
            "quantity_required",
            "quantity_reserved",
        ]
        read_only_fields = fields


class WorkOrderSerializer(serializers.ModelSerializer):
    parts = WorkOrderPartSerializer(many=True, read_only=True)

    class Meta:
        model = WorkOrder
        fields = [
            "id",
            "bearing",
            "alert",
            "status",
            "priority",
            "scheduled_start",
            "scheduled_end",
            "actual_start",
            "actual_end",
            "assigned_technician",
            "notes",
            "created_at",
            "parts",
        ]
        # api.md Write Restrictions: `status` is never directly writable via
        # PATCH — the transition/ action is the only mutator. This viewset
        # additionally offers no create/update at all (read-only viewset),
        # so read_only_fields here is belt-and-suspenders documentation.
        read_only_fields = ["id", "status", "created_at", "parts"]


class WorkOrderTransitionSerializer(serializers.Serializer):
    """
    Input-only serializer for POST /work-orders/{id}/transition/.

    Field requiredness genuinely depends on `to_status` (scheduling.md §4),
    which DRF's static `required=` can't express — so every field here is
    optional at the serializer level, and `apps.scheduling.transitions`
    enforces the real per-transition requirement, returning 400 with a
    specific message when something's missing.
    """
    to_status = serializers.ChoiceField(choices=WorkOrder.Status.choices)
    scheduled_start = serializers.DateTimeField(required=False)
    scheduled_end = serializers.DateTimeField(required=False)
    notes = serializers.CharField(required=False, allow_blank=True)
