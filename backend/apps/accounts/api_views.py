from django.contrib.auth import get_user_model
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.pagination import CursorPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.inventory.permissions import IsAdminRole
from apps.registry.mixins import ProtectedDeleteMixin
from config.api_errors import Conflict, InvalidQueryParam
from config.query_params import BadParam, parse_bool, parse_enum

from .api_serializers import UserSerializer

User = get_user_model()


class UserCursorPagination(CursorPagination):
    ordering = "username"  # unik -> cursor stabil


class UserViewSet(ProtectedDeleteMixin, viewsets.ModelViewSet):
    """
    api.md: Users — read/write ADMIN. Ekstensi kecil: GET /users/me/ (semua role
    terautentikasi) agar frontend tahu role pengguna setelah login (JWT tidak
    memuat role). DELETE: tidak bisa menghapus akun sendiri; user yang masih
    direferensikan FK PROTECT (mis. export_jobs) -> 409 RFC7807.
    """

    queryset = User.objects.all()
    serializer_class = UserSerializer
    pagination_class = UserCursorPagination

    def get_permissions(self):
        if self.action == "me":
            return [IsAuthenticated()]
        return [IsAdminRole()]

    def list(self, request, *args, **kwargs):
        params = request.query_params
        try:
            roles = parse_enum(params, "role", set(User.Role.values))
            active = parse_bool(params, "is_active")
        except BadParam as exc:
            raise InvalidQueryParam(str(exc))
        qs = self.get_queryset()
        if roles:
            qs = qs.filter(role__in=roles)
        if active is not None:
            qs = qs.filter(is_active=active)
        page = self.paginate_queryset(qs)
        if page is not None:
            return self.get_paginated_response(self.get_serializer(page, many=True).data)
        return Response(self.get_serializer(qs, many=True).data)

    @action(detail=False, methods=["get"], url_path="me")
    def me(self, request):
        return Response(UserSerializer(request.user).data)

    def perform_destroy(self, instance):
        if instance.pk == self.request.user.pk:
            raise Conflict("Tidak bisa menghapus akun sendiri.")
        instance.delete()
