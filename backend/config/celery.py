"""
Celery app entrypoint. Four separate queues per architecture.md
§Why Separate Celery Queues — never a shared default queue, so a backlog
in one stage (e.g. exports) cannot delay the real-time ingestion→alert
chain (signal_processing → ml_inference → alerting).
"""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

app = Celery("predictmaint")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

app.conf.task_queues = {
    "signal_processing": {"exchange": "signal_processing", "routing_key": "signal_processing"},
    "ml_inference": {"exchange": "ml_inference", "routing_key": "ml_inference"},
    "alerting": {"exchange": "alerting", "routing_key": "alerting"},
    "export": {"exchange": "export", "routing_key": "export"},
}

app.conf.task_routes = {
    "apps.signal_processing.tasks.*": {"queue": "signal_processing"},
    "apps.ml_inference.tasks.*": {"queue": "ml_inference"},
    "apps.alerting.tasks.*": {"queue": "alerting"},
    "apps.exports.tasks.*": {"queue": "export"},
}

app.conf.task_default_queue = "signal_processing"
