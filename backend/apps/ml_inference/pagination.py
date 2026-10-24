from rest_framework.pagination import CursorPagination


class PredictionCursorPagination(CursorPagination):
    """api.md: predictions tumbuh terus -> cursor, bukan offset. Ordering eksplisit."""
    # Urutan sekunder "-id" WAJIB: tanpa itu baris ber-timestamp sama berurutan acak antar halaman
    # (duplikat/terlewat). DRF memakai ordering[0] sebagai posisi cursor; sisanya menstabilkan urutan.
    ordering = ("-predicted_at", "-id")
