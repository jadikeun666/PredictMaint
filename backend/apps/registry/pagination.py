"""
Pagination untuk registry — api.md §Conventions: CursorPagination.

CursorPagination DRF punya default `ordering = "-created"` (placeholder
generik). Machine punya `created_at`; Bearing & Sensor TIDAK punya kolom
timestamp sama sekali, jadi dipakai `-id` (UUID valid sebagai cursor,
urutannya stabil walau bukan kronologis).
"""
from rest_framework.pagination import CursorPagination


class CreatedAtCursorPagination(CursorPagination):
    # Urutan sekunder "-id" WAJIB: tanpa itu baris ber-timestamp sama berurutan acak antar halaman
    # (duplikat/terlewat). DRF memakai ordering[0] sebagai posisi cursor; sisanya menstabilkan urutan.
    ordering = ("-created_at", "-id")


class IdCursorPagination(CursorPagination):
    ordering = "-id"
