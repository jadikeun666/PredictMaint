"""
DIAGNOSTIC_PDF — exports.md §2. Satu bearing, rentang tanggal terpilih.

Struktur tetap: (1) header, (2) panel ringkasan (prediksi terbaru "per saat laporan dibuat",
tidak dibatasi rentang), (3) tabel riwayat fault (CLASSIFICATION != HEALTHY dalam rentang),
(4) grafik tren RUL (query InfluxDB yang SAMA dengan dashboard: query_predictions_trend),
(5) alert & work order terkait, (6) catatan model_version yang menghasilkan angka di laporan.

Definisi "dalam rentang" (start inklusif, end eksklusif): prediksi -> predicted_at,
alert -> triggered_at, work order -> created_at (konsisten dengan HISTORY_EXCEL).
Import berat (weasyprint, matplotlib) lazy di dalam fungsi.
"""
import base64
import io
import logging
import os
from datetime import timezone as dt_timezone

from django.template.loader import render_to_string
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from apps.alerting.models import Alert
from apps.ml_inference.models import Prediction
from apps.registry.models import Bearing
from apps.scheduling.models import WorkOrder
from config.api_errors import ServiceUnavailable
from config.influx_queries import MAX_LIMIT, query_predictions_trend

from .storage import output_path, partial_path

logger = logging.getLogger(__name__)

TEMPLATE = "exports/diagnostic_report.html"
MAX_TABLE_ROWS = 2000  # batas baris per tabel; total sebenarnya selalu ditampilkan


def fmt_dt(dt):
    if dt is None:
        return "—"
    return dt.astimezone(dt_timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_rul_chart(points):
    """PNG (data URI) predicted_rul_hours terhadap waktu, atau None bila tak ada titik."""
    pairs = [(p["predicted_at"], p["predicted_rul_hours"]) for p in points if p.get("predicted_rul_hours") is not None]
    if not pairs:
        return None
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure  # tanpa pyplot: tidak ada state global

    fig = Figure(figsize=(7.2, 2.9), dpi=150)
    ax = fig.subplots()
    ax.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", markersize=2.5, linewidth=1.2, color="#1f6feb")
    ax.set_ylabel("RUL prediksi (jam)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _resolution(alert):
    if alert.resolved_at is None:
        return "Belum resolved"
    who = alert.resolved_by.username if alert.resolved_by_id else "sistem"
    return f"Resolved {fmt_dt(alert.resolved_at)} oleh {who}"


def build_context(job, now=None):
    params = job.params
    start, end = parse_datetime(params["start"]), parse_datetime(params["end"])
    bearing = Bearing.objects.select_related("machine").get(pk=params["bearing_id"])
    generated_at = now or timezone.now()
    ptype = Prediction.PredictionType

    base = Prediction.objects.filter(bearing=bearing)
    cls_all = base.filter(prediction_type=ptype.CLASSIFICATION)
    cls_in_range = cls_all.filter(predicted_at__gte=start, predicted_at__lt=end)

    # (2) ringkasan: terbaru per saat laporan dibuat (tidak dibatasi rentang)
    latest_cls = cls_all.filter(predicted_at__lte=generated_at).order_by("-predicted_at", "-id").first()
    latest_rul = base.filter(prediction_type=ptype.RUL, predicted_at__lte=generated_at).order_by("-predicted_at", "-id").first()

    # (3) riwayat fault
    faults = cls_in_range.exclude(fault_class=Prediction.FaultClass.HEALTHY)
    fault_total = faults.count()
    fault_rows = [
        {
            "at": fmt_dt(p.predicted_at),
            "fault_class": p.fault_class,
            "confidence": f"{p.classification_confidence:.3f}" if p.classification_confidence is not None else "—",
        }
        for p in faults.order_by("-predicted_at", "-id")[:MAX_TABLE_ROWS]
    ]

    # (4) tren RUL — query yang sama dengan dashboard; InfluxDB mati -> PDF tetap jadi + catatan
    rul_points, rul_truncated, rul_error = [], False, None
    try:
        rul_points, rul_truncated = query_predictions_trend(
            bearing_id=bearing.id, start=start, end=end, limit=MAX_LIMIT, prediction_type="RUL"
        )
    except ServiceUnavailable:
        rul_error = "Data tren RUL tidak dapat diambil (InfluxDB tidak terjangkau saat laporan dibuat)."
        logger.warning("Export %s: grafik RUL dilewati karena InfluxDB tidak terjangkau", job.id)
    chart = render_rul_chart(rul_points)

    # (5) alert & work order terkait
    alerts_qs = (
        Alert.objects.filter(bearing=bearing, triggered_at__gte=start, triggered_at__lt=end)
        .select_related("resolved_by").order_by("triggered_at", "id")
    )
    wo_qs = WorkOrder.objects.filter(bearing=bearing, created_at__gte=start, created_at__lt=end).order_by("created_at", "id")
    alert_total, wo_total = alerts_qs.count(), wo_qs.count()
    alert_rows = [
        {"alert_type": a.alert_type, "severity": a.severity, "triggered_at": fmt_dt(a.triggered_at),
         "last_seen_at": fmt_dt(a.last_seen_at), "resolution": _resolution(a)}
        for a in alerts_qs[:MAX_TABLE_ROWS]
    ]
    wo_rows = [
        {"short_id": str(w.id)[:8], "status": w.status, "priority": w.priority,
         "scheduled": f"{fmt_dt(w.scheduled_start)} s/d {fmt_dt(w.scheduled_end)}" if w.scheduled_start else "—",
         "actual_end": fmt_dt(w.actual_end), "notes": (w.notes or "")[:140]}
        for w in wo_qs[:MAX_TABLE_ROWS]
    ]

    # (6) provenance: semua model_version di balik angka pada laporan
    versions = set(cls_in_range.values_list("model_version", flat=True).distinct())
    versions.update(p.model_version for p in (latest_cls, latest_rul) if p)
    versions.update(pt["model_version"] for pt in rul_points if pt.get("model_version"))

    def pred_row(p, kind):
        if p is None:
            return None
        if kind == "cls":
            return {"fault_class": p.fault_class, "confidence": f"{p.classification_confidence:.3f}",
                    "at": fmt_dt(p.predicted_at), "model_version": p.model_version}
        f = lambda v: f"{v:.2f}" if v is not None else "—"
        return {"hours": f(p.predicted_rul_hours), "low": f(p.rul_confidence_band_low_hours),
                "high": f(p.rul_confidence_band_high_hours), "at": fmt_dt(p.predicted_at),
                "model_version": p.model_version}

    return {
        "bearing": bearing,
        "machine": bearing.machine,
        "start_text": fmt_dt(start),
        "end_text": fmt_dt(end),
        "generated_text": fmt_dt(generated_at),
        "generated_by": job.requested_by.username,
        "latest_cls": pred_row(latest_cls, "cls"),
        "latest_rul": pred_row(latest_rul, "rul"),
        "class_total": cls_in_range.count(),
        "fault_total": fault_total,
        "fault_rows": fault_rows,
        "fault_truncated": fault_total > len(fault_rows),
        "chart": chart,
        "rul_count": len(rul_points),
        "rul_truncated": rul_truncated,
        "rul_error": rul_error,
        "alert_total": alert_total,
        "alert_rows": alert_rows,
        "alert_truncated": alert_total > len(alert_rows),
        "wo_total": wo_total,
        "wo_rows": wo_rows,
        "wo_truncated": wo_total > len(wo_rows),
        "model_versions": sorted(versions),
        "max_rows": MAX_TABLE_ROWS,
    }


def render_html(context):
    return render_to_string(TEMPLATE, context)


def build_diagnostic_pdf(job):
    from weasyprint import HTML  # lazy: import berat

    html = render_html(build_context(job))
    final = output_path(job)
    tmp = partial_path(final)
    HTML(string=html).write_pdf(str(tmp))
    tmp.replace(final)  # rename atomik: file final tidak pernah setengah jadi
    logger.info("DIAGNOSTIC_PDF %s -> %s", job.id, final)
    return final
