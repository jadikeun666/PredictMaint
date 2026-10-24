"""
Penyimpanan file export — exports.md §4: "storage yang sama dengan raw waveform"
(volume lokal di dev, object storage di deployment skala besar). File ditulis worker
`export` ke <RAW_WAVEFORM_STORAGE_PATH>/exports/<job_id>.<ext>; django-api membacanya
dari volume yang sama untuk endpoint unduh.
"""
from pathlib import Path

from django.conf import settings

EXPORT_SUBDIR = "exports"

CONTENT_TYPES = {
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
EXTENSIONS = {"DIAGNOSTIC_PDF": "pdf", "HISTORY_EXCEL": "xlsx"}


def export_dir() -> Path:
    path = Path(settings.RAW_WAVEFORM_STORAGE_PATH) / EXPORT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(job) -> Path:
    return export_dir() / f"{job.id}.{EXTENSIONS[job.export_type]}"


def partial_path(final: Path) -> Path:
    """File sementara; di-rename atomik ke final setelah selesai ditulis."""
    return final.with_name(final.name + ".part")


def resolve_download_path(job):
    """Path file yang aman untuk disajikan, atau None bila tidak ada / di luar folder export."""
    if not job.file_path:
        return None
    root = export_dir().resolve()
    path = Path(job.file_path).resolve()
    if root not in path.parents or not path.is_file():
        return None
    return path
