from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from config.query_params import (
    BadParam,
    multi,
    parse_dt,
    parse_enum,
    parse_uuid_list,
    problem_400,
)

from .models import Prediction
from .pagination import PredictionCursorPagination
from .serializers import PredictionSerializer


class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    """
    api.md: Predictions — read = semua role terautentikasi, tanpa endpoint tulis
    (system-only). ReadOnlyModelViewSet -> POST/PUT/PATCH/DELETE = 405
    (api.md §Write Restrictions).
    """

    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = PredictionCursorPagination

    def _filtered(self, params):
        qs = self.get_queryset()
        bearings = parse_uuid_list(params, "bearing")
        types = parse_enum(params, "prediction_type", set(Prediction.PredictionType.values))
        classes = parse_enum(params, "fault_class", set(Prediction.FaultClass.values))
        versions = multi(params, "model_version")
        start = parse_dt(params, "start")
        end = parse_dt(params, "end")
        if start and end and start >= end:
            raise BadParam("'start' harus lebih awal dari 'end'.")
        if bearings:
            qs = qs.filter(bearing_id__in=bearings)
        if types:
            qs = qs.filter(prediction_type__in=types)
        if classes:
            qs = qs.filter(fault_class__in=classes)
        if versions:
            qs = qs.filter(model_version__in=versions)
        if start:
            qs = qs.filter(predicted_at__gte=start)   # start inklusif
        if end:
            qs = qs.filter(predicted_at__lt=end)      # end eksklusif
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
