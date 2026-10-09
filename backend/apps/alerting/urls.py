from django.urls import include, path
from rest_framework.routers import SimpleRouter

from .views import AlertViewSet

router = SimpleRouter()
router.register(r"alerts", AlertViewSet, basename="alert")

urlpatterns = [
    path("", include(router.urls)),
]
