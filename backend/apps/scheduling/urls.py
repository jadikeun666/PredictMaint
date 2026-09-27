from rest_framework.routers import DefaultRouter

from .views import WorkOrderViewSet

router = DefaultRouter()
router.register("work-orders", WorkOrderViewSet, basename="work-order")

urlpatterns = router.urls
