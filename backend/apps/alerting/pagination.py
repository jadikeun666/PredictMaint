from rest_framework.pagination import CursorPagination


class AlertCursorPagination(CursorPagination):
    """Ordering eksplisit (default DRF '-created' tidak ada di model ini)."""
    # Urutan sekunder "-id" WAJIB: tanpa itu baris ber-timestamp sama berurutan acak antar halaman
    # (duplikat/terlewat). DRF memakai ordering[0] sebagai posisi cursor; sisanya menstabilkan urutan.
    ordering = ("-triggered_at", "-id")
