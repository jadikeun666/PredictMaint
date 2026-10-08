from rest_framework import viewsets

from .mixins import ProtectedDeleteMixin, RegistryPermissionMixin
from .models import Bearing, Machine, Sensor
from .pagination import CreatedAtCursorPagination, IdCursorPagination
from .serializers import BearingSerializer, MachineSerializer, SensorSerializer


class MachineViewSet(
    RegistryPermissionMixin, ProtectedDeleteMixin, viewsets.ModelViewSet
):
    """api.md: Machines — read = all, write (incl. delete) = ADMIN."""

    queryset = Machine.objects.all()
    serializer_class = MachineSerializer
    pagination_class = CreatedAtCursorPagination


class BearingViewSet(
    RegistryPermissionMixin, ProtectedDeleteMixin, viewsets.ModelViewSet
):
    """
    api.md: Bearings — read = all, write (incl. delete) = ADMIN.
    Geometri (d, D, φ, n) nullable saat create; diisi belakangan via PATCH
    (wajib sebelum BPFO/BPFI/BSF/FTF bisa dihitung — prd.md A2).
    """

    queryset = Bearing.objects.select_related("machine").all()
    serializer_class = BearingSerializer
    pagination_class = IdCursorPagination


class SensorViewSet(
    RegistryPermissionMixin, ProtectedDeleteMixin, viewsets.ModelViewSet
):
    """api.md: Sensors — read = all, write (incl. delete) = ADMIN. mqtt_topic auto-generated."""

    queryset = Sensor.objects.select_related("bearing__machine").all()
    serializer_class = SensorSerializer
    pagination_class = IdCursorPagination
