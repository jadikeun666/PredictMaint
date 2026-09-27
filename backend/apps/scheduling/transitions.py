"""
Work Order status state machine — scheduling.md §4.

Giliran 1: transition validity, role gate, required-fields, non-stock
side effects (actual_start/actual_end, scheduled_start/end, notes).

Giliran 2 added inventory.md §2 stock effects:
- DRAFT -> SCHEDULED: reserve stock per work_order_parts row (§2 poin 1).
- IN_PROGRESS -> COMPLETED: StockMovement ISSUE + quantity_on_hand
  decrement, atomic (§2 poin 2); linked alert resolved if not already
  (scheduling.md §4 "Side effects" column for this transition).
- Any transition to CANCELLED: release reservation, no StockMovement
  row (§2 poin 3).

Giliran 3 (this version) wires in inventory.md §4: after the ISSUE
movement created below, call apps.inventory.services.check_low_stock so
the "after ANY stock_movements insert" trigger condition is honored for
ISSUE too, not just the RECEIPT/ADJUSTMENT paths in apps.inventory.views.
That module is the single place the LOW_STOCK check lives — this file
calls into it rather than duplicating the cooldown/creation logic.
"""
from django.db import transaction
from django.utils import timezone

from apps.accounts.models import User
from apps.alerting.models import Alert
from apps.inventory.models import StockMovement
from apps.inventory.services import check_low_stock

from .models import WorkOrder

Status = WorkOrder.Status
Role = User.Role

# scheduling.md §4 "Who" column, translated to Role enum members.
TRANSITIONS = {
    (Status.DRAFT, Status.SCHEDULED): {
        "roles": {Role.ENGINEER, Role.ADMIN},
        "required_fields": ("scheduled_start", "scheduled_end"),
    },
    (Status.DRAFT, Status.CANCELLED): {
        "roles": {Role.ENGINEER, Role.ADMIN},
        "required_fields": (),
    },
    (Status.SCHEDULED, Status.CANCELLED): {
        "roles": {Role.ENGINEER, Role.ADMIN},
        "required_fields": (),
    },
    (Status.SCHEDULED, Status.IN_PROGRESS): {
        "roles": {Role.TECHNICIAN, Role.ENGINEER, Role.ADMIN},
        "required_fields": (),
    },
    (Status.IN_PROGRESS, Status.COMPLETED): {
        "roles": {Role.TECHNICIAN, Role.ENGINEER, Role.ADMIN},
        "required_fields": (),
    },
    (Status.IN_PROGRESS, Status.CANCELLED): {
        # scheduling.md §4: ENGINEER, ADMIN only — NOT TECHNICIAN.
        "roles": {Role.ENGINEER, Role.ADMIN},
        "required_fields": ("notes",),
    },
}

# inventory.md §2 poin 1: exact warning text format.
INSUFFICIENT_STOCK_NOTE = "Insufficient stock for {sku}: have {on_hand}, need {required}"


class TransitionError(Exception):
    """Base class; the view maps each subclass to an HTTP status."""


class InvalidTransition(TransitionError):
    """(from, to) pair not in TRANSITIONS -> HTTP 409 per api.md."""


class Forbidden(TransitionError):
    """Role not permitted for this specific transition -> HTTP 403."""


class MissingField(TransitionError):
    """Required field absent/blank for this transition -> HTTP 400."""


def get_rule(from_status, to_status):
    rule = TRANSITIONS.get((from_status, to_status))
    if rule is None:
        raise InvalidTransition(f"{from_status} -> {to_status} is not a valid transition")
    return rule


def _reserve_stock(work_order):
    """
    inventory.md §2 poin 1 (DRAFT -> SCHEDULED). For each work_order_parts
    row: sufficient on-hand -> quantity_reserved = quantity_required (soft
    hold, quantity_on_hand NOT decremented yet). Insufficient -> leave
    quantity_reserved at 0 and return a warning line for the caller to
    append to notes; the transition still succeeds either way.
    """
    warnings = []
    parts = list(work_order.parts.select_related("spare_part").select_for_update())
    for part in parts:
        spare = part.spare_part
        if spare.quantity_on_hand >= part.quantity_required:
            part.quantity_reserved = part.quantity_required
            part.save(update_fields=["quantity_reserved"])
        else:
            warnings.append(
                INSUFFICIENT_STOCK_NOTE.format(
                    sku=spare.sku,
                    on_hand=spare.quantity_on_hand,
                    required=part.quantity_required,
                )
            )
    return warnings


def _issue_reserved_stock(work_order, now):
    """
    inventory.md §2 poin 2 (IN_PROGRESS -> COMPLETED). For each part with
    quantity_reserved > 0: create a StockMovement ISSUE row and decrement
    quantity_on_hand by the same amount. Caller (apply_transition) wraps
    this in transaction.atomic so the movement + decrement never happen
    one without the other.
    """
    parts = list(work_order.parts.select_related("spare_part").select_for_update())
    for part in parts:
        if part.quantity_reserved <= 0:
            continue
        spare = part.spare_part
        StockMovement.objects.create(
            spare_part=spare,
            movement_type=StockMovement.MovementType.ISSUE,
            quantity=-part.quantity_reserved,
            reference_work_order=work_order,
            timestamp=now,
        )
        spare.quantity_on_hand = spare.quantity_on_hand - part.quantity_reserved
        spare.save(update_fields=["quantity_on_hand"])
        check_low_stock(spare, now)


def _release_reserved_stock(work_order):
    """
    inventory.md §2 poin 3 (any transition to CANCELLED). Reset
    quantity_reserved to 0 for every part — NO StockMovement row, since a
    reservation was never a physical stock movement.
    """
    work_order.parts.filter(quantity_reserved__gt=0).update(quantity_reserved=0)


def _resolve_linked_alert(work_order, now):
    """
    scheduling.md §4 Side effects column, IN_PROGRESS -> COMPLETED:
    "linked alerts.resolved_at set if not already resolved." System-
    resolved (resolved_by=null), same pattern as SENSOR_OFFLINE/LOW_STOCK
    auto-resolve elsewhere in this codebase.
    """
    if work_order.alert_id:
        Alert.objects.filter(pk=work_order.alert_id, resolved_at__isnull=True).update(
            resolved_at=now, resolved_by=None
        )


@transaction.atomic
def apply_transition(work_order, to_status, user, data, now=None):
    """
    Validate + apply a single transition, including inventory side
    effects, all in one DB transaction. Raises TransitionError subclasses
    on failure (nothing is written if validation fails, since the whole
    function is wrapped in transaction.atomic). Returns the mutated,
    saved work_order on success.
    """
    now = now or timezone.now()
    from_status = work_order.status
    rule = get_rule(from_status, to_status)

    if user.role not in rule["roles"]:
        raise Forbidden(
            f"Role {user.role} is not permitted to transition {from_status} -> {to_status}"
        )

    for field in rule["required_fields"]:
        value = data.get(field)
        if value in (None, ""):
            raise MissingField(f"'{field}' is required for {from_status} -> {to_status}")

    stock_warnings = []

    if to_status == Status.SCHEDULED:
        work_order.scheduled_start = data["scheduled_start"]
        work_order.scheduled_end = data["scheduled_end"]
        stock_warnings = _reserve_stock(work_order)
    elif to_status == Status.IN_PROGRESS:
        work_order.actual_start = now
    elif to_status == Status.COMPLETED:
        work_order.actual_end = now
        _issue_reserved_stock(work_order, now)
        _resolve_linked_alert(work_order, now)
    elif to_status == Status.CANCELLED:
        notes = data.get("notes")
        if notes:
            work_order.notes = f"{work_order.notes}\n{notes}".strip() if work_order.notes else notes
        _release_reserved_stock(work_order)

    if stock_warnings:
        combined = "\n".join(stock_warnings)
        work_order.notes = f"{work_order.notes}\n{combined}".strip() if work_order.notes else combined

    work_order.status = to_status
    work_order.save()
    return work_order
