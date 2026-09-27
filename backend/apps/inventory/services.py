"""
Low-stock alerting — inventory.md §4.

Deliberately shared, not duplicated: every StockMovement-creating path
(manual RECEIPT/ADJUSTMENT via apps.inventory.views, and the ISSUE
created by apps.scheduling.transitions on Work Order completion) calls
into this module so "after ANY stock_movements insert, re-check
quantity_on_hand < reorder_point" is enforced from exactly one place.
"""
from apps.alerting.models import Alert


def check_low_stock(spare_part, now):
    """
    inventory.md §4 trigger condition. Call after every StockMovement
    insert, once spare_part.quantity_on_hand already reflects it.

    - Below reorder_point + no open LOW_STOCK alert for this part ->
      create one (severity=WARNING, bearing=null, spare_part_id set).
    - Below reorder_point + an open LOW_STOCK alert already exists ->
      update last_seen_at only (cooldown — same pattern as bearing-health
      alerts, scheduling.md §1 / prd.md D1).
    - At/above reorder_point -> no-op here; resolving an existing alert
      is resolve_low_stock_if_replenished's job, called only on RECEIPT
      per the spec's literal wording (see that function's docstring).
    """
    if spare_part.quantity_on_hand >= spare_part.reorder_point:
        return

    existing = Alert.objects.filter(
        alert_type=Alert.AlertType.LOW_STOCK,
        spare_part_id=spare_part.id,
        resolved_at__isnull=True,
    ).first()

    if existing:
        existing.last_seen_at = now
        existing.save(update_fields=["last_seen_at"])
    else:
        Alert.objects.create(
            bearing=None,
            spare_part=spare_part,
            alert_type=Alert.AlertType.LOW_STOCK,
            severity=Alert.Severity.WARNING,
            triggered_at=now,
            last_seen_at=now,
        )


def resolve_low_stock_if_replenished(spare_part, now):
    """
    inventory.md §4 Resolution: "when a subsequent RECEIPT movement
    brings quantity_on_hand >= reorder_point, the open LOW_STOCK alert
    for that part is auto-resolved." Taken literally — this is called
    only from the RECEIPT path, not from ADJUSTMENT or ISSUE, since the
    spec names RECEIPT specifically for this side effect.

    System-resolved (resolved_by=null) — same pattern as SENSOR_OFFLINE/
    SENSOR_CONFIG_MISSING auto-resolve elsewhere in this codebase.
    """
    if spare_part.quantity_on_hand < spare_part.reorder_point:
        return
    Alert.objects.filter(
        alert_type=Alert.AlertType.LOW_STOCK,
        spare_part_id=spare_part.id,
        resolved_at__isnull=True,
    ).update(resolved_at=now, resolved_by=None)
