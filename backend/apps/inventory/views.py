from django.db import transaction
from django.utils import timezone
from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated

from .models import SparePart, StockMovement
from .pagination import SparePartCursorPagination, StockMovementCursorPagination
from .permissions import IsAdminRole, IsEngineerOrAdminRole
from .serializers import SparePartSerializer, StockMovementSerializer
from .services import check_low_stock, resolve_low_stock_if_replenished


class SparePartViewSet(viewsets.ModelViewSet):
    """
    api.md: Spare Parts read = all authenticated roles; write
    (create/edit) = ADMIN only. No destroy endpoint is exposed — deleting
    a spare part that has movement/compatibility history would orphan
    those FKs (both use on_delete=PROTECT), so this session does not add
    a delete action; ADMIN can still deactivate via other means if ever
    needed (out of scope here).
    """

    queryset = SparePart.objects.all().order_by("sku")
    serializer_class = SparePartSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = SparePartCursorPagination
    http_method_names = ["get", "post", "put", "patch", "head", "options"]

    def get_permissions(self):
        if self.action in ("create", "update", "partial_update"):
            return [IsAuthenticated(), IsAdminRole()]
        return [IsAuthenticated()]


class StockMovementViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    """
    api.md: Stock Movements read = all authenticated roles; write =
    ENGINEER, ADMIN. Deliberately create/list/retrieve only — movements
    are an append-only ledger (like vibration_windows/predictions
    elsewhere in this codebase): no update or delete action is exposed,
    since editing history after the fact would desync quantity_on_hand
    from its own audit trail.
    """

    queryset = StockMovement.objects.all().order_by("-timestamp")
    serializer_class = StockMovementSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = StockMovementCursorPagination

    def get_permissions(self):
        if self.action == "create":
            return [IsAuthenticated(), IsEngineerOrAdminRole()]
        return [IsAuthenticated()]

    @transaction.atomic
    def perform_create(self, serializer):
        now = timezone.now()
        movement = serializer.save(timestamp=now)

        # Lock the row for the read-modify-write below — StockMovement
        # insert + quantity_on_hand update must be atomic (inventory.md
        # §3/§6: "in the same transaction").
        spare = SparePart.objects.select_for_update().get(pk=movement.spare_part_id)
        spare.quantity_on_hand = spare.quantity_on_hand + movement.quantity
        spare.save(update_fields=["quantity_on_hand"])

        if movement.movement_type == StockMovement.MovementType.RECEIPT:
            resolve_low_stock_if_replenished(spare, now)
        check_low_stock(spare, now)
