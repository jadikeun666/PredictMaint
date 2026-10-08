from django.urls import include, path
from rest_framework.routers import SimpleRouter

from .views import BearingViewSet, MachineViewSet, SensorViewSet

router = SimpleRouter()
router.register(r"machines", MachineViewSet, basename="machine")
router.register(r"bearings", BearingViewSet, basename="bearing")
router.register(r"sensors", SensorViewSet, basename="sensor")

urlpatterns = [
    path("", include(router.urls)),
]
