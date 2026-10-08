"""
URL configuration for config project.

api.md §Conventions: base path /api/v1/, auth via
Authorization: Bearer <JWT> on every endpoint except /api/v1/auth/token/.
"""
from django.contrib import admin
from django.urls import include, path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/v1/auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/v1/", include("apps.scheduling.urls")),
    path("api/v1/", include("apps.inventory.urls")),
    path("api/v1/", include("apps.registry.urls")),
]
