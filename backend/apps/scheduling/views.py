from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import WorkOrder
from .pagination import WorkOrderCursorPagination
from .serializers import WorkOrderSerializer, WorkOrderTransitionSerializer
from .transitions import Forbidden, InvalidTransition, MissingField, apply_transition


def problem(status_code, title, detail):
    """RFC 7807-style problem JSON — api.md §Conventions."""
    return Response(
        {"type": "about:blank", "title": title, "status": status_code, "detail": detail},
        status=status_code,
    )


class WorkOrderViewSet(viewsets.ReadOnlyModelViewSet):
    """
    api.md: Work Orders read = all authenticated roles.

    Deliberately read-only at the ModelViewSet level (list/retrieve only,
    no create/update/destroy) — this session's scope is the transition/
    action per OBJEKTIF poin 1; full manual Work Order creation
    (scheduling.md §5) is not part of this session.
    """

    queryset = WorkOrder.objects.all().order_by("-created_at")
    serializer_class = WorkOrderSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = WorkOrderCursorPagination

    @action(detail=True, methods=["post"], url_path="transition")
    def transition(self, request, pk=None):
        work_order = self.get_object()
        serializer = WorkOrderTransitionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = dict(serializer.validated_data)
        to_status = data.pop("to_status")

        try:
            apply_transition(work_order, to_status, request.user, data)
        except InvalidTransition as exc:
            return problem(409, "Invalid transition", str(exc))
        except Forbidden as exc:
            return problem(403, "Forbidden", str(exc))
        except MissingField as exc:
            return problem(400, "Missing field", str(exc))

        return Response(WorkOrderSerializer(work_order).data)
