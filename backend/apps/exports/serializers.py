from datetime import timezone as dt_timezone

from rest_framework import serializers

from apps.registry.models import Bearing, Machine

from .models import ExportJob


def _iso(dt):
    return dt.astimezone(dt_timezone.utc).isoformat().replace("+00:00", "Z")


class _WindowParams(serializers.Serializer):
    start = serializers.DateTimeField()
    end = serializers.DateTimeField()

    def validate(self, attrs):
        if attrs["start"] >= attrs["end"]:
            raise serializers.ValidationError({"end": "Harus lebih akhir dari 'start'."})
        return attrs


class DiagnosticPdfParams(_WindowParams):
    bearing_id = serializers.PrimaryKeyRelatedField(queryset=Bearing.objects.all())


class HistoryExcelParams(_WindowParams):
    machine_ids = serializers.ListField(
        child=serializers.PrimaryKeyRelatedField(queryset=Machine.objects.all()),
        required=False, max_length=500,
    )
    bearing_ids = serializers.ListField(
        child=serializers.PrimaryKeyRelatedField(queryset=Bearing.objects.all()),
        required=False, max_length=500,
    )


PARAM_SERIALIZERS = {
    ExportJob.ExportType.DIAGNOSTIC_PDF: DiagnosticPdfParams,
    ExportJob.ExportType.HISTORY_EXCEL: HistoryExcelParams,
}


class ExportJobCreateSerializer(serializers.Serializer):
    """
    exports.md §1: tepat dua export_type. `params` divalidasi KETAT per tipe (parameter tak
    dikenal ditolak, tidak diabaikan diam-diam) lalu disimpan ternormalisasi (ISO8601 UTC, UUID string).
    """

    export_type = serializers.ChoiceField(choices=ExportJob.ExportType.choices)
    params = serializers.DictField()

    def validate(self, attrs):
        raw = attrs["params"]
        param_cls = PARAM_SERIALIZERS[attrs["export_type"]]
        errors = {
            key: "Parameter tidak dikenal untuk export_type ini."
            for key in sorted(set(raw) - set(param_cls().fields))
        }
        inner = param_cls(data=raw)
        if not inner.is_valid():
            errors.update(inner.errors)
        if errors:
            raise serializers.ValidationError({"params": errors})

        data = inner.validated_data
        normalized = {"start": _iso(data["start"]), "end": _iso(data["end"])}
        if "bearing_id" in data:
            normalized["bearing_id"] = str(data["bearing_id"].pk)
        for key in ("machine_ids", "bearing_ids"):
            if data.get(key):
                normalized[key] = sorted({str(obj.pk) for obj in data[key]})
        attrs["params"] = normalized
        return attrs


class ExportJobSerializer(serializers.ModelSerializer):
    """
    Nama field mengikuti database.md. `file_path` (path internal server) sengaja tidak
    diekspos; diganti `download_url` yang terisi hanya saat COMPLETED.
    """

    download_url = serializers.SerializerMethodField()

    class Meta:
        model = ExportJob
        fields = [
            "id",
            "export_type",
            "requested_by",
            "params",
            "status",
            "requested_at",
            "completed_at",
            "download_url",
        ]
        read_only_fields = fields

    def get_download_url(self, obj):
        if obj.status == ExportJob.Status.COMPLETED and obj.file_path:
            return f"/api/v1/export-jobs/{obj.id}/download/"
        return None
