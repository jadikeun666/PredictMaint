from django.utils import timezone
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.inventory.permissions import IsEngineerOrAdminRole
from config.query_params import BadParam, parse_bool, parse_enum, parse_uuid_list, problem_400

from .models import Alert
from .pagination import AlertCursorPagination
from .serializers import AlertSerializer


class AlertViewSet(viewsets.ReadOnlyModelViewSet):
    """
    api.md: Alerts — read = semua role terautentikasi; resolve = ENGINEER, ADMIN.
    Tidak ada create/update/delete (alert dibuat sistem). POST /resolve/ tidak
    cascade ke Work Order (api.md) — status WO independen.
    """

    queryset = Alert.objects.all()
    serializer_class = AlertSerializer
    pagination_class = AlertCursorPagination

    def get_permissions(self):
        if self.action == "resolve":
            return [IsEngineerOrAdminRole()]
        return [IsAuthenticated()]

    def _filtered(self, params):
        qs = self.get_queryset()
        bearings = parse_uuid_list(params, "bearing")
        types = parse_enum(params, "alert_type", set(Alert.AlertType.values))
        severities = parse_enum(params, "severity", set(Alert.Severity.values))
        resolved = parse_bool(params, "resolved")
        if bearings:
            qs = qs.filter(bearing_id__in=bearings)
        if types:
            qs = qs.filter(alert_type__in=types)
        if severities:
            qs = qs.filter(severity__in=severities)
        if resolved is not None:
            qs = qs.filter(resolved_at__isnull=not resolved)
        return qs

    def list(self, request, *args, **kwargs):
        try:
            qs = self._filtered(request.query_params)
        except BadParam as exc:
            return problem_400(str(exc))
        page = self.paginate_queryset(qs)
        if page is not None:
            return self.get_paginated_response(self.get_serializer(page, many=True).data)
        return Response(self.get_serializer(qs, many=True).data)

    @action(detail=True, methods=["post"], url_path="resolve")
    def resolve(self, request, pk=None):
        alert = self.get_object()
        # Update bersyarat: aman terhadap dua resolve bersamaan (hanya satu yang menang).
        updated = Alert.objects.filter(pk=alert.pk, resolved_at__isnull=True).update(
            resolved_at=timezone.now(), resolved_by=request.user
        )
        if not updated:
            return Response(
                {
                    "type": "about:blank",
                    "title": "Conflict",
                    "status": 409,
                    "detail": "Alert ini sudah resolved.",
                },
                status=409,
            )
        alert.refresh_from_db()
        return Response(AlertSerializer(alert).data)
