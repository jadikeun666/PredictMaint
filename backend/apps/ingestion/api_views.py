from rest_framework import viewsets
from rest_framework.pagination import CursorPagination
from rest_framework.response import Response

from apps.inventory.permissions import IsEngineerOrAdminRole
from config.api_errors import InvalidQueryParam
from config.query_params import BadParam, parse_dt, parse_enum, parse_uuid_list

from .api_serializers import VibrationWindowSerializer
from .models import VibrationWindow


class VibrationWindowCursorPagination(CursorPagination):
    # Urutan sekunder "-id" WAJIB: tanpa itu baris ber-timestamp sama berurutan acak antar halaman
    # (duplikat/terlewat). DRF memakai ordering[0] sebagai posisi cursor; sisanya menstabilkan urutan.
    ordering = ("-captured_at", "-id")


class VibrationWindowViewSet(viewsets.ReadOnlyModelViewSet):
    """
    api.md: Vibration Windows — read ENGINEER, ADMIN; tanpa endpoint tulis
    (immutable, system-only -> POST/PUT/PATCH/DELETE = 405).
    """

    queryset = VibrationWindow.objects.all()
    serializer_class = VibrationWindowSerializer
    permission_classes = [IsEngineerOrAdminRole]
    pagination_class = VibrationWindowCursorPagination

    def _filtered(self, params):
        qs = self.get_queryset()
        sensors = parse_uuid_list(params, "sensor")
        bearings = parse_uuid_list(params, "bearing")
        datasets = parse_enum(params, "source_dataset", set(VibrationWindow.SourceDataset.values))
        start = parse_dt(params, "start")
        end = parse_dt(params, "end")
        if start and end and start >= end:
            raise BadParam("'start' harus lebih awal dari 'end'.")
        if sensors:
            qs = qs.filter(sensor_id__in=sensors)
        if bearings:
            qs = qs.filter(sensor__bearing_id__in=bearings)
        if datasets:
            qs = qs.filter(source_dataset__in=datasets)
        if start:
            qs = qs.filter(captured_at__gte=start)  # inklusif
        if end:
            qs = qs.filter(captured_at__lt=end)     # eksklusif
        return qs

    def list(self, request, *args, **kwargs):
        try:
            qs = self._filtered(request.query_params)
        except BadParam as exc:
            raise InvalidQueryParam(str(exc))
        page = self.paginate_queryset(qs)
        if page is not None:
            return self.get_paginated_response(self.get_serializer(page, many=True).data)
        return Response(self.get_serializer(qs, many=True).data)
