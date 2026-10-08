from django.urls import path

from .gantt_views import GanttView

urlpatterns = [
    path("gantt/", GanttView.as_view(), name="gantt"),
]
