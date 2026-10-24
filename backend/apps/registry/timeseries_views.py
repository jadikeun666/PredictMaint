from datetime import timedelta

from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from config.api_errors import InvalidQueryParam
from config.influx_queries import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    PREDICTION_TYPES,
    query_derived_features,
    query_predictions_trend,
)
from config.query_params import BadParam, parse_dt, parse_enum

from .models import Bearing

DEFAULT_WINDOW = timedelta(hours=24)


def _parse_window(params):
    """start inklusif, end eksklusif (konsisten dengan predictions/vibration-windows)."""
    end = parse_dt(params, "end") or timezone.now()
    start = parse_dt(params, "start") or end - DEFAULT_WINDOW
    if start >= end:
        raise BadParam("'start' harus lebih awal dari 'end'.")
    raw = params.get("limit")
    if raw in (None, ""):
        limit = DEFAULT_LIMIT
    else:
        try:
            limit = int(raw)
        except ValueError:
            raise BadParam(f"'limit' harus bilangan bulat, bukan {raw!r}.")
        if not 1 <= limit <= MAX_LIMIT:
            raise BadParam(f"'limit' harus antara 1 dan {MAX_LIMIT}.")
    return start, end, limit


class DerivedFeaturesView(APIView):
    """
    api.md: GET /bearings/{id}/derived-features/ — membaca InfluxDB `derived_features`.
    Semua role terautentikasi. Default rentang = 24 jam terakhir.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        bearing = get_object_or_404(Bearing, pk=pk)
        try:
            start, end, limit = _parse_window(request.query_params)
        except BadParam as exc:
            raise InvalidQueryParam(str(exc))
        points, truncated = query_derived_features(
            bearing_id=bearing.id, start=start, end=end, limit=limit
        )
        return Response(
            {
                "bearing_id": bearing.id,
                "start": start,
                "end": end,
                "limit": limit,
                "count": len(points),
                "truncated": truncated,
                "points": points,
            }
        )


class PredictionsTrendView(APIView):
    """
    api.md: GET /bearings/{id}/predictions-trend/ — membaca InfluxDB `predictions_ts`.
    Param tambahan: prediction_type (CLASSIFICATION|RUL). Semua role terautentikasi.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        bearing = get_object_or_404(Bearing, pk=pk)
        try:
            start, end, limit = _parse_window(request.query_params)
            types = parse_enum(request.query_params, "prediction_type", set(PREDICTION_TYPES))
            if len(types) > 1:
                raise BadParam("'prediction_type' hanya boleh satu nilai.")
        except BadParam as exc:
            raise InvalidQueryParam(str(exc))
        ptype = types[0] if types else None
        points, truncated = query_predictions_trend(
            bearing_id=bearing.id, start=start, end=end, limit=limit, prediction_type=ptype
        )
        return Response(
            {
                "bearing_id": bearing.id,
                "prediction_type": ptype,
                "start": start,
                "end": end,
                "limit": limit,
                "count": len(points),
                "truncated": truncated,
                "points": points,
            }
        )
