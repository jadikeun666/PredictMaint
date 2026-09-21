"""
Draft Work Order auto-creation — scheduling.md §2-§3, prd.md D2, inventory.md §1.

Giliran 2: WorkOrder DRAFT + WorkOrderPart dari compatibility lookup.
Giliran 3: suggested scheduled_start/end (scheduling.md §3).
Belum: reservasi stock (inventory.md §2) -> sesi terpisah.
"System proposes, human decides": WorkOrder SELALU status DRAFT; jadwal hanya
saran yang boleh diubah manusia saat DRAFT -> SCHEDULED.
"""
from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from apps.alerting.models import Alert
from apps.inventory.models import BearingPartCompatibility
from apps.registry.models import Bearing

from .conflicts import find_earliest_free_block, machine_active_intervals
from .models import WorkOrder, WorkOrderPart

# scheduling.md §2: HIGH severity -> priority HIGH, CRITICAL -> URGENT.
PRIORITY_BY_SEVERITY = {
    Alert.Severity.HIGH.value: WorkOrder.Priority.HIGH.value,
    Alert.Severity.CRITICAL.value: WorkOrder.Priority.URGENT.value,
}

# scheduling.md §3 (default fixed)
MAINTENANCE_BLOCK = timedelta(hours=4)
TARGET_DEADLINE = {
    WorkOrder.Priority.URGENT.value: timedelta(hours=48),
    WorkOrder.Priority.HIGH.value: timedelta(days=7),
}

# Teks persis dari spec
NO_COMPATIBLE_PART_NOTE = (
    "No compatible spare part registered for bearing_model={model} — add via Inventory admin"
)
NO_FREE_WINDOW_NOTE = (
    "Scheduling conflict — no free maintenance window before target deadline, manual scheduling required"
)

DEFAULT_QUANTITY_REQUIRED = 1  # inventory.md §2: best-guess 1 unit per compatible part


def suggest_schedule_window(machine_id, priority, now):
    """
    scheduling.md §3. Return (start, end, has_free_slot).
    Tidak ada slot bebas sebelum deadline -> saran tetap di awal window target
    (start = now), TIDAK didorong keluar deadline, has_free_slot=False.
    Interpretasi "target deadline's start" = awal window target (now).
    Sistem tidak pernah menggeser Work Order lain.
    """
    deadline = now + TARGET_DEADLINE[priority]
    busy = machine_active_intervals(machine_id, not_before=now)
    start = find_earliest_free_block(busy, now, deadline, MAINTENANCE_BLOCK)
    if start is None:
        return now, now + MAINTENANCE_BLOCK, False
    return start, start + MAINTENANCE_BLOCK, True


@transaction.atomic
def create_draft_work_order(alert, now=None):
    """
    Return WorkOrder DRAFT untuk `alert`, atau None kalau severity tidak
    memicu Work Order / alert bukan berbasis bearing. Idempotent per alert.
    `now` hanya untuk pengujian; default waktu sekarang.
    """
    priority = PRIORITY_BY_SEVERITY.get(str(alert.severity))
    if priority is None or alert.bearing_id is None:
        return None

    existing = WorkOrder.objects.filter(alert_id=alert.id).first()
    if existing is not None:
        return existing

    now = now or timezone.now()
    bearing = Bearing.objects.get(pk=alert.bearing_id)

    part_ids = list(
        BearingPartCompatibility.objects.filter(bearing_model=bearing.bearing_model)
        .order_by("spare_part_id")
        .values_list("spare_part_id", flat=True)
        .distinct()
    )
    notes = []
    if not part_ids:
        notes.append(NO_COMPATIBLE_PART_NOTE.format(model=bearing.bearing_model))

    start, end, has_slot = suggest_schedule_window(bearing.machine_id, priority, now)
    if not has_slot:
        notes.append(NO_FREE_WINDOW_NOTE)

    work_order = WorkOrder.objects.create(
        bearing_id=alert.bearing_id,
        alert=alert,
        status=WorkOrder.Status.DRAFT,
        priority=priority,
        scheduled_start=start,
        scheduled_end=end,
        notes="\n".join(notes),
    )
    WorkOrderPart.objects.bulk_create(
        [
            WorkOrderPart(
                work_order=work_order,
                spare_part_id=part_id,
                quantity_required=DEFAULT_QUANTITY_REQUIRED,
                quantity_reserved=0,
            )
            for part_id in part_ids
        ]
    )
    return work_order
