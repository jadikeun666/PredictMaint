from django.urls import include, path
from rest_framework.routers import SimpleRouter

from .api_views import UserViewSet

router = SimpleRouter()
router.register(r"users", UserViewSet, basename="user")

urlpatterns = [
    path("", include(router.urls)),
]
