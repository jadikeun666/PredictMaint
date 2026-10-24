from django.urls import path

from .timeseries_views import DerivedFeaturesView, PredictionsTrendView

urlpatterns = [
    path("bearings/<uuid:pk>/derived-features/", DerivedFeaturesView.as_view(), name="bearing-derived-features"),
    path("bearings/<uuid:pk>/predictions-trend/", PredictionsTrendView.as_view(), name="bearing-predictions-trend"),
]
