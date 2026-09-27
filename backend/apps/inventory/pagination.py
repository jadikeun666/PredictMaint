from rest_framework.pagination import CursorPagination


class SparePartCursorPagination(CursorPagination):
    """SparePart has no timestamp column; sku is unique, so it's a valid
    cursor ordering field."""
    ordering = "sku"


class StockMovementCursorPagination(CursorPagination):
    ordering = "-timestamp"
