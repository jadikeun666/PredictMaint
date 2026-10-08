"""
GET /api/v1/gantt/ — gantt.md §2-§5, api.md "GET /gantt/".

Read-only, denormalized: work_orders (SCHEDULED/IN_PROGRESS) + bearings +
machines, satu baris per mesin, flag konflik dihitung di server.
Aturan overlap SATU-SATUNYA ada di conflicts.intervals_overlap (diimpor).
"""
import uuid
from datetime import datetime, timedelta, timezone as dt_timezone

from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.registry.models import Machine

from .conflicts import ACTIVE_STATUSES, intervals_overlap
from .models import WorkOrder

DEFAULT_RANGE_DAYS = 14  # gantt.md §5: current date -> +14 days
# Urutan LOW < MEDIUM < HIGH < URGENT diturunkan dari model (gantt.md §3: ordering tetap).
PRIORITY_RANK = {value: i for i, (value, _) in enumerate(WorkOrder.Priority.choices)}


class BadParam(Exception):
    pass


def _problem(detail):
    return Response(
        {"type": "about:blank", "title": "Bad request", "status": 400, "detail": detail},
        status=400,
    )


def _multi(params, name):
    """Terima ?name=a&name=b maupun ?name=a,b."""
    values = []
    for raw in params.getlist(name):
        values.extend(v.strip() for v in raw.split(",") if v.strip())
    return values


def _parse_enum(params, name, allowed):
    values = _multi(params, name)
    bad = [v for v in values if v not in allowed]
    if bad:
        raise BadParam(f"Nilai '{name}' tidak valid: {bad}. Pilihan: {sorted(allowed)}.")
    return set(values) if values else set(allowed)


def _parse_machine_ids(params):
    ids = []
    for v in _multi(params, "machine"):
        try:
            ids.append(uuid.UUID(v))
        except ValueError:
            raise BadParam(f"'machine' bukan UUID valid: {v!r}.")
    return ids


def _parse_dt(raw, name):
    try:
        dt = parse_datetime(raw)
    except ValueError:
        dt = None
    if dt is None:
        raise BadParam(
            f"'{name}' bukan ISO8601 valid: {raw!r}. Gunakan format seperti "
            "2026-09-26T00:00:00Z (tanda '+' di URL harus ditulis %2B)."
        )
    if timezone.is_naive(dt):
        dt = dt.replace(tzinfo=dt_timezone.utc)
    return dt


def _parse_range(params):
    raw_start, raw_end = params.get("start"), params.get("end")
    if raw_start:
        start = _parse_dt(raw_start, "start")
    else:
        today = timezone.localtime(timezone.now()).replace(hour=0, minute=0, second=0, microsecond=0)
        start = today
    end = _parse_dt(raw_end, "end") if raw_end else start + timedelta(days=DEFAULT_RANGE_DAYS)
    if start >= end:
        raise BadParam("'start' harus lebih awal dari 'end'.")
    return start, end


def _technician(user):
    if user is None:
        return None
    full = f"{user.first_name} {user.last_name}".strip()
    return {"id": user.id, "name": full or user.username}


class GanttView(APIView):
    permission_classes = [IsAuthenticated]  # read = semua role; hanya GET diimplementasikan

    def get(self, request):
        params = request.query_params
        try:
            machine_ids = _parse_machine_ids(params)
            priorities = _parse_enum(params, "priority", set(PRIORITY_RANK))
            statuses = _parse_enum(params, "status", {s.value for s in ACTIVE_STATUSES})
            range_start, range_end = _parse_range(params)
        except BadParam as exc:
            return _problem(str(exc))

        machines = Machine.objects.order_by("name", "id")
        if machine_ids:
            machines = machines.filter(id__in=machine_ids)
        machines = list(machines)

        # Semua WO aktif pada mesin terpilih — TANPA filter status/priority/rentang —
        # karena konflik adalah fakta jadwal, bukan fungsi filter tampilan.
        active = (
            WorkOrder.objects.filter(
                bearing__machine__in=machines,
                status__in=ACTIVE_STATUSES,
                scheduled_start__isnull=False,
                scheduled_end__isnull=False,
            )
            .select_related("bearing", "assigned_technician")
            .order_by("scheduled_start", "id")
        )
        by_machine = {}
        for wo in active:
            by_machine.setdefault(wo.bearing.machine_id, []).append(wo)

        rows = []
        for m in machines:
            group = by_machine.get(m.id, [])
            bars = []
            for wo in group:
                if wo.status not in statuses or wo.priority not in priorities:
                    continue
                if not intervals_overlap(wo.scheduled_start, wo.scheduled_end, range_start, range_end):
                    continue
                conflicts = [
                    o.id
                    for o in group
                    if o.id != wo.id
                    and intervals_overlap(wo.scheduled_start, wo.scheduled_end, o.scheduled_start, o.scheduled_end)
                ]
                tech = _technician(wo.assigned_technician)
                label = wo.bearing.position_label + (f" — {tech['name']}" if tech else "")
                bars.append(
                    {
                        "id": wo.id,
                        "bearing_id": wo.bearing_id,
                        "bearing_position_label": wo.bearing.position_label,
                        "status": wo.status,
                        "priority": wo.priority,
                        "priority_rank": PRIORITY_RANK[wo.priority],
                        "scheduled_start": wo.scheduled_start,
                        "scheduled_end": wo.scheduled_end,
                        "assigned_technician": tech,
                        "label": label,
                        "in_conflict": bool(conflicts),
                        "conflicts_with": conflicts,
                    }
                )
            rows.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "criticality": m.criticality,
                    "work_orders": bars,
                }
            )

        return Response(
            {"range": {"start": range_start, "end": range_end}, "machines": rows}
        )
