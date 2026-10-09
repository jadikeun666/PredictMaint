from rest_framework.pagination import CursorPagination


class AlertCursorPagination(CursorPagination):
    """Ordering eksplisit (default DRF '-created' tidak ada di model ini)."""
    ordering = "-triggered_at"
