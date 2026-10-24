from django.urls import include, path
from rest_framework.routers import SimpleRouter

from .api_views import VibrationWindowViewSet

router = SimpleRouter()
router.register(r"vibration-windows", VibrationWindowViewSet, basename="vibration-window")

urlpatterns = [
    path("", include(router.urls)),
]
