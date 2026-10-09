from rest_framework.pagination import CursorPagination


class PredictionCursorPagination(CursorPagination):
    """api.md: predictions tumbuh terus -> cursor, bukan offset. Ordering eksplisit."""
    ordering = "-predicted_at"
