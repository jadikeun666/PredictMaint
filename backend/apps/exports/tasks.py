"""
Task Celery queue `export` (exports.md §4). Import berat (openpyxl, weasyprint, matplotlib)
SENGAJA lazy di dalam fungsi: config/celery.py autodiscover mengimpor tasks.py di SEMUA worker
(architecture.md §Operational Lesson), jadi import tingkat modul di sini akan memengaruhi
worker lain.
"""
import logging

from celery import shared_task
from django.utils import timezone

from .models import ExportJob
from .storage import output_path, partial_path  # noqa: F401  (partial_path dipakai generator)

logger = logging.getLogger(__name__)


def _generate(job):
    if job.export_type == ExportJob.ExportType.HISTORY_EXCEL:
        from .excel_report import build_history_excel

        return build_history_excel(job)
    if job.export_type == ExportJob.ExportType.DIAGNOSTIC_PDF:
        from .pdf_report import build_diagnostic_pdf

        return build_diagnostic_pdf(job)
    raise ValueError(f"export_type tidak dikenal: {job.export_type}")


@shared_task
def run_export(job_id):
    """PENDING -> RUNNING -> COMPLETED | FAILED. Alasan gagal hanya di log (exports.md §4.3)."""
    # Klaim atomik: hanya satu worker yang memproses job PENDING (aman terhadap redelivery).
    claimed = ExportJob.objects.filter(pk=job_id, status=ExportJob.Status.PENDING).update(
        status=ExportJob.Status.RUNNING
    )
    if not claimed:
        logger.warning("run_export(%s) dilewati: job tidak ada atau bukan PENDING", job_id)
        return
    job = ExportJob.objects.get(pk=job_id)
    try:
        path = _generate(job)
    except Exception:
        logger.exception("Export job %s (%s) GAGAL", job_id, job.export_type)
        ExportJob.objects.filter(pk=job_id).update(status=ExportJob.Status.FAILED)
        return
    ExportJob.objects.filter(pk=job_id).update(
        status=ExportJob.Status.COMPLETED, file_path=str(path), completed_at=timezone.now()
    )
    logger.info("Export job %s selesai: %s", job_id, path)
