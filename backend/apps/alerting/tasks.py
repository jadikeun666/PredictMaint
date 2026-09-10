"""
Celery tasks for 'alerting'. No task logic yet — implementation is a later
session per claude.md §Next Session build order. This file exists so
autodiscover_tasks() and task_routes in config/celery.py resolve cleanly.
"""
from celery import shared_task
