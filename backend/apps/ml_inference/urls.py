from django.urls import include, path
from rest_framework.routers import SimpleRouter

from .views import PredictionViewSet

router = SimpleRouter()
router.register(r"predictions", PredictionViewSet, basename="prediction")

urlpatterns = [
    path("", include(router.urls)),
]
