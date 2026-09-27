"""
Role-gate permission classes — engineering-rules.md §Architecture Rules:
"one permission class per role-gated action, reused across viewsets",
not scattered `if request.user.role == ...` checks in views.
"""
from rest_framework.permissions import BasePermission

from apps.accounts.models import User


class IsAdminRole(BasePermission):
    """api.md: Spare Parts write (create/edit) = ADMIN only."""

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.role == User.Role.ADMIN
        )


class IsEngineerOrAdminRole(BasePermission):
    """api.md: Stock Movements write = ENGINEER, ADMIN."""

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.role in (User.Role.ENGINEER, User.Role.ADMIN)
        )
