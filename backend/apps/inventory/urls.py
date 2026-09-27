from rest_framework.routers import DefaultRouter

from .views import SparePartViewSet, StockMovementViewSet

router = DefaultRouter()
router.register("spare-parts", SparePartViewSet, basename="spare-part")
router.register("stock-movements", StockMovementViewSet, basename="stock-movement")

urlpatterns = router.urls
