from rest_framework import serializers

from .models import SparePart, StockMovement


class SparePartSerializer(serializers.ModelSerializer):
    class Meta:
        model = SparePart
        fields = [
            "id",
            "sku",
            "name",
            "quantity_on_hand",
            "reorder_point",
            "reorder_quantity",
            "unit_cost",
            "supplier",
        ]
        read_only_fields = ["id"]

    def get_fields(self):
        fields = super().get_fields()
        if self.instance is not None:
            # Existing part (PUT/PATCH) — quantity_on_hand changes ONLY
            # via a StockMovement (RECEIPT, ISSUE, ADJUSTMENT), never a
            # direct write here (api.md §Write Restrictions pattern,
            # applied to this field). On CREATE, though, this is the only
            # way to set a new part's starting stock — there is no prior
            # movement history to derive it from — so it stays writable
            # and required there (self.instance is None means create).
            fields["quantity_on_hand"].read_only = True
        return fields


class StockMovementSerializer(serializers.ModelSerializer):
    spare_part_sku = serializers.CharField(source="spare_part.sku", read_only=True)

    class Meta:
        model = StockMovement
        fields = [
            "id",
            "spare_part",
            "spare_part_sku",
            "movement_type",
            "quantity",
            "reference_work_order",
            "timestamp",
            "reason",
        ]
        # reference_work_order is set only by the system (ISSUE on Work
        # Order completion, apps.scheduling.transitions) — never via this
        # endpoint, since ISSUE itself is rejected here (see validate()).
        # timestamp is set server-side in the view, not client-supplied.
        read_only_fields = ["id", "spare_part_sku", "reference_work_order", "timestamp"]

    def validate(self, attrs):
        movement_type = attrs.get("movement_type")
        quantity = attrs.get("quantity")

        if movement_type == StockMovement.MovementType.ISSUE:
            raise serializers.ValidationError(
                {
                    "movement_type": (
                        "ISSUE movements are system-only — created automatically "
                        "when a Work Order is completed (scheduling.md §4), not "
                        "via this endpoint."
                    )
                }
            )

        if movement_type == StockMovement.MovementType.RECEIPT and quantity is not None and quantity <= 0:
            raise serializers.ValidationError(
                {"quantity": "RECEIPT quantity must be positive (database.md)."}
            )

        if movement_type == StockMovement.MovementType.ADJUSTMENT and not attrs.get("reason"):
            raise serializers.ValidationError(
                {"reason": "reason is required for every ADJUSTMENT movement (inventory.md §6)."}
            )

        return attrs
