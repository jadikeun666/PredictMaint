from rest_framework.pagination import CursorPagination


class WorkOrderCursorPagination(CursorPagination):
    """
    DRF's base CursorPagination has `ordering = "-created"` as a class
    default — a placeholder meant to be overridden per view, NOT derived
    from the queryset's own .order_by(). WorkOrder has no `created` field
    (it's `created_at`, per database.md), so the base default 500s on
    every list call regardless of what .order_by() the viewset's queryset
    uses. This subclass fixes the actual field name.
    """
    ordering = "-created_at"
