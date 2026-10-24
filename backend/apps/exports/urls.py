from django.urls import include, path
from rest_framework.routers import SimpleRouter

from .views import ExportJobViewSet

router = SimpleRouter()
router.register(r"export-jobs", ExportJobViewSet, basename="export-job")

urlpatterns = [
    path("", include(router.urls)),
]
