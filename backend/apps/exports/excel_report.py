"""
HISTORY_EXCEL — exports.md §3. Workbook multi-sheet; header kolom = nama field persis
database.md (atribut konkret model, mis. `bearing_id`) tanpa penamaan ulang.

Sheet: Predictions, Alerts, WorkOrders, Inventory.
Definisi "dalam rentang" (keputusan implementasi; start inklusif, end eksklusif):
  Predictions -> predicted_at | Alerts -> triggered_at | WorkOrders -> created_at |
  Inventory: spare_parts = snapshot saat export (tanpa rentang), stock_movements -> timestamp.
Filter mesin/bearing: bila diberikan, hanya baris ber-bearing di cakupan itu (alert tanpa bearing,
mis. LOW_STOCK, tidak ikut); tanpa filter semua baris ikut.
Mode write-only (hemat memori untuk export besar).
"""
import json
import logging
import uuid
from datetime import datetime, timezone as dt_timezone
from decimal import Decimal

from django.utils import timezone
from django.utils.dateparse import parse_datetime
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font

from apps.alerting.models import Alert
from apps.inventory.models import SparePart, StockMovement
from apps.ml_inference.models import Prediction
from apps.registry.models import Bearing
from apps.scheduling.models import WorkOrder

from .storage import output_path, partial_path

logger = logging.getLogger(__name__)
BOLD = Font(bold=True)


def _value(v):
    if v is None or isinstance(v, (bool, int, float, str, Decimal)):
        return v
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, datetime):  # Excel tidak mendukung timezone -> UTC naif
        return v.astimezone(dt_timezone.utc).replace(tzinfo=None) if timezone.is_aware(v) else v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


def _cell(ws, v):
    v = _value(v)
    if isinstance(v, str) and v.startswith("="):  # cegah injeksi formula dari teks bebas
        c = WriteOnlyCell(ws, value=v)
        c.data_type = "s"
        return c
    return v


def _bold(ws, text):
    c = WriteOnlyCell(ws, value=text)
    c.font = BOLD
    return c


def _write_block(ws, queryset, model):
    fields = list(model._meta.concrete_fields)
    ws.append([_bold(ws, f.attname) for f in fields])
    count = 0
    for obj in queryset.iterator(chunk_size=2000):
        ws.append([_cell(ws, getattr(obj, f.attname)) for f in fields])
        count += 1
    return count


def _new_sheet(wb, title, widths_for=None):
    ws = wb.create_sheet(title)
    ws.freeze_panes = "A2"
    return ws


def _bounds(params):
    return parse_datetime(params["start"]), parse_datetime(params["end"])


def build_history_excel(job):
    params = job.params
    start, end = _bounds(params)
    machine_ids = params.get("machine_ids") or []
    bearing_ids = params.get("bearing_ids") or []
    scoped = bool(machine_ids or bearing_ids)
    bearing_scope = Bearing.objects.all()
    if machine_ids:
        bearing_scope = bearing_scope.filter(machine_id__in=machine_ids)
    if bearing_ids:
        bearing_scope = bearing_scope.filter(id__in=bearing_ids)

    def scope(qs):
        return qs.filter(bearing_id__in=bearing_scope.values("id")) if scoped else qs

    wb = Workbook(write_only=True)

    ws = _new_sheet(wb, "Predictions")
    n_pred = _write_block(
        ws,
        scope(Prediction.objects.filter(predicted_at__gte=start, predicted_at__lt=end)).order_by("predicted_at", "id"),
        Prediction,
    )

    ws = _new_sheet(wb, "Alerts")
    n_alert = _write_block(
        ws,
        scope(Alert.objects.filter(triggered_at__gte=start, triggered_at__lt=end)).order_by("triggered_at", "id"),
        Alert,
    )

    ws = _new_sheet(wb, "WorkOrders")  # semua status (berbeda dari Gantt, exports.md §3)
    n_wo = _write_block(
        ws,
        scope(WorkOrder.objects.filter(created_at__gte=start, created_at__lt=end)).order_by("created_at", "id"),
        WorkOrder,
    )

    ws = _new_sheet(wb, "Inventory")
    ws.append([_bold(ws, f"Spare parts (snapshot saat export: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')} UTC)")])
    n_parts = _write_block(ws, SparePart.objects.order_by("sku", "id"), SparePart)
    ws.append([])
    ws.append([_bold(ws, f"Stock movements ({params['start']} s/d {params['end']}, end eksklusif)")])
    n_mov = _write_block(
        ws,
        StockMovement.objects.filter(timestamp__gte=start, timestamp__lt=end).order_by("timestamp", "id"),
        StockMovement,
    )

    final = output_path(job)
    tmp = partial_path(final)
    wb.save(tmp)
    tmp.replace(final)  # rename atomik: file final tidak pernah setengah jadi
    logger.info(
        "HISTORY_EXCEL %s: predictions=%d alerts=%d work_orders=%d spare_parts=%d stock_movements=%d",
        job.id, n_pred, n_alert, n_wo, n_parts, n_mov,
    )
    return final
