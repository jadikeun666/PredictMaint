"""
Aturan konflik jadwal — gantt.md §4 (SATU-SATUNYA definisi aturan overlap).

Dua Work Order konflik jika dan hanya jika: mesin (machine_id) sama DAN
interval [scheduled_start, scheduled_end] beririsan:
    A.start < B.end AND B.start < A.end
Dipakai oleh suggested-schedule (scheduling.md §3). View Gantt (gantt.md §4)
harus mengimpor intervals_overlap dari sini, bukan menulis ulang aturannya.
"""
from .models import WorkOrder

# Hanya Work Order berstatus ini yang "menempati" jadwal mesin (scheduling.md §3).
ACTIVE_STATUSES = (WorkOrder.Status.SCHEDULED, WorkOrder.Status.IN_PROGRESS)


def intervals_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def find_earliest_free_block(busy, earliest, deadline, block):
    """
    Cari blok `block` (timedelta) paling awal yang mulai >= `earliest`, selesai
    <= `deadline`, dan tidak beririsan dengan interval mana pun di `busy`
    (iterable (start, end)). Return start blok, atau None kalau tidak ada.
    Satu kali sapuan cukup: busy diurutkan menurut start, kandidat hanya maju.
    """
    candidate = earliest
    for b_start, b_end in sorted(busy):
        if intervals_overlap(candidate, candidate + block, b_start, b_end):
            candidate = b_end
    if candidate + block <= deadline:
        return candidate
    return None


def machine_active_intervals(machine_id, not_before):
    """Interval Work Order SCHEDULED/IN_PROGRESS di satu mesin yang belum lewat."""
    return list(
        WorkOrder.objects.filter(
            bearing__machine_id=machine_id,
            status__in=ACTIVE_STATUSES,
            scheduled_start__isnull=False,
            scheduled_end__isnull=False,
            scheduled_end__gt=not_before,  # pruning interval yang sudah lewat, bukan aturan overlap
        ).values_list("scheduled_start", "scheduled_end")
    )
