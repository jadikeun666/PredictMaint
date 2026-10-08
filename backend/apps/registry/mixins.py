"""
Mixin bersama untuk viewset registry (machines/bearings/sensors).

- RegistryPermissionMixin: read = semua user terautentikasi,
  write (create/update/partial_update/destroy) = ADMIN (api.md).
  Mengimpor IsAdminRole dari apps.inventory.permissions (tidak ditulis ulang).
- ProtectedDeleteMixin: DELETE pada row yang masih direferensikan FK
  on_delete=PROTECT menghasilkan RFC7807 409, bukan HTTP 500.
"""
from collections import Counter

from django.db.models import ProtectedError
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.inventory.permissions import IsAdminRole


class RegistryPermissionMixin:
    def get_permissions(self):
        if self.action in ("list", "retrieve"):
            return [IsAuthenticated()]
        return [IsAdminRole()]


class ProtectedDeleteMixin:
    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        try:
            self.perform_destroy(instance)
        except ProtectedError as exc:
            counts = Counter(obj._meta.label for obj in exc.protected_objects)
            refs = ", ".join(f"{label} ({n})" for label, n in sorted(counts.items()))
            return Response(
                {
                    "type": "about:blank",
                    "title": "Conflict",
                    "status": 409,
                    "detail": (
                        "Tidak bisa dihapus: masih direferensikan oleh "
                        f"{refs}. Hapus/pindahkan data terkait terlebih dahulu."
                    ),
                },
                status=status.HTTP_409_CONFLICT,
                content_type="application/problem+json",
            )
        return Response(status=status.HTTP_204_NO_CONTENT)
