"""
Pagination untuk registry — api.md §Conventions: CursorPagination.

CursorPagination DRF punya default `ordering = "-created"` (placeholder
generik). Machine punya `created_at`; Bearing & Sensor TIDAK punya kolom
timestamp sama sekali, jadi dipakai `-id` (UUID valid sebagai cursor,
urutannya stabil walau bukan kronologis).
"""
from rest_framework.pagination import CursorPagination


class CreatedAtCursorPagination(CursorPagination):
    ordering = "-created_at"


class IdCursorPagination(CursorPagination):
    ordering = "-id"
