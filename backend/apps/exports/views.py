import logging

from django.http import FileResponse
from rest_framework import mixins, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import NotFound
from rest_framework.pagination import CursorPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.inventory.permissions import IsEngineerOrAdminRole
from config.api_errors import Conflict, InvalidQueryParam, ServiceUnavailable
from config.celery import app as celery_app
from config.query_params import BadParam, parse_enum

from .models import ExportJob
from .serializers import ExportJobCreateSerializer, ExportJobSerializer
from .storage import CONTENT_TYPES, EXTENSIONS, resolve_download_path

logger = logging.getLogger(__name__)

TASK_NAME = "apps.exports.tasks.run_export"
EXPORT_QUEUE = "export"  # terpisah dari signal_processing/ml_inference (exports.md §4)


class ExportJobCursorPagination(CursorPagination):
    ordering = ("-requested_at", "-id")


class ExportJobViewSet(
    mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet
):
    """
    api.md/exports.md: POST -> 202 + id; polling GET /export-jobs/{id}/; unduh via
    GET /export-jobs/{id}/download/. Tulis (POST) = ENGINEER, ADMIN untuk KEDUA tipe
    (mengikuti tabel api.md); baca/unduh = semua role terautentikasi.
    Tidak ada update/delete (job dikelola sistem; file tidak dihapus otomatis di v1).
    """

    queryset = ExportJob.objects.all()
    pagination_class = ExportJobCursorPagination

    def get_permissions(self):
        if self.action == "create":
            return [IsEngineerOrAdminRole()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        return ExportJobCreateSerializer if self.action == "create" else ExportJobSerializer

    def list(self, request, *args, **kwargs):
        params = request.query_params
        try:
            statuses = parse_enum(params, "status", set(ExportJob.Status.values))
            types = parse_enum(params, "export_type", set(ExportJob.ExportType.values))
        except BadParam as exc:
            raise InvalidQueryParam(str(exc))
        qs = self.get_queryset()
        if statuses:
            qs = qs.filter(status__in=statuses)
        if types:
            qs = qs.filter(export_type__in=types)
        page = self.paginate_queryset(qs)
        if page is not None:
            return self.get_paginated_response(self.get_serializer(page, many=True).data)
        return Response(self.get_serializer(qs, many=True).data)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = ExportJob.objects.create(
            export_type=serializer.validated_data["export_type"],
            requested_by=request.user,
            params=serializer.validated_data["params"],
        )
        try:
            celery_app.send_task(TASK_NAME, args=[str(job.id)], queue=EXPORT_QUEUE)
        except Exception as exc:
            logger.exception("Gagal mengantrekan export job %s", job.id)
            ExportJob.objects.filter(pk=job.pk).update(status=ExportJob.Status.FAILED)
            raise ServiceUnavailable("Antrian export tidak tersedia; coba lagi.") from exc
        return Response(
            ExportJobSerializer(job).data,
            status=202,
            headers={"Location": f"/api/v1/export-jobs/{job.id}/"},
        )

    @action(detail=True, methods=["get"], url_path="download")
    def download(self, request, pk=None):
        job = self.get_object()
        if job.status != ExportJob.Status.COMPLETED:
            raise Conflict(f"Export belum siap diunduh (status={job.status}).")
        path = resolve_download_path(job)
        if path is None:
            raise NotFound("File export tidak ditemukan di penyimpanan.")
        ext = EXTENSIONS[job.export_type]
        return FileResponse(
            open(path, "rb"),
            as_attachment=True,
            filename=f"{job.export_type.lower()}-{job.id}.{ext}",
            content_type=CONTENT_TYPES[ext],
        )
