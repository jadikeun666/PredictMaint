"""
Regresi Sesi F: pesan status OFFLINE (LWT / retained sisa shutdown) tidak boleh
dianggap "sensor kembali online" dan me-resolve alert SENSOR_OFFLINE.
Handler dipanggil langsung (tanpa broker).
"""
from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from apps.alerting.models import Alert
from apps.ingestion.management.commands.run_ingestion import Command, OFFLINE_THRESHOLD_S
from apps.ingestion.services import upsert_sensor_offline_alert
from apps.registry.models import Bearing, Machine, Sensor

TOPIC = "factory/test-rig/drive-end/status"
VIB_TOPIC = "factory/test-rig/drive-end/vibration"


def _payload(status):
    return {"status": status, "rpm": 1772, "last_seen": "2026-11-11T08:00:00.000000Z"}


class StatusHandlerTests(TestCase):
    def setUp(self):
        m = Machine.objects.create(name="Test Rig", location="lab", machine_type="rig", criticality="MEDIUM")
        self.bearing = Bearing.objects.create(
            machine=m, position_label="drive-end", bearing_model="X", status="ACTIVE")
        self.sensor = Sensor.objects.create(
            bearing=self.bearing, sensor_type="ACCELEROMETER", axis="RADIAL",
            sample_rate_hz=12000, source="SIMULATED", mqtt_topic=VIB_TOPIC)
        self.cmd = Command()
        self.cmd._last_status_seen = {}

    def _handle(self, status):
        self.cmd._handle_status(TOPIC, _payload(status), b"{}")

    def _active_offline(self):
        return Alert.objects.filter(
            bearing=self.bearing, alert_type=Alert.AlertType.SENSOR_OFFLINE, resolved_at__isnull=True)

    def test_offline_status_does_not_resolve_active_alert(self):
        alert, _ = upsert_sensor_offline_alert(self.sensor)
        self._handle("OFFLINE")
        alert.refresh_from_db()
        self.assertIsNone(alert.resolved_at)

    def test_online_status_resolves_active_alert_by_system(self):
        alert, _ = upsert_sensor_offline_alert(self.sensor)
        self._handle("ONLINE")
        alert.refresh_from_db()
        self.assertIsNotNone(alert.resolved_at)
        self.assertIsNone(alert.resolved_by_id)

    def test_offline_status_is_tracked_and_detector_updates_existing_alert(self):
        alert, _ = upsert_sensor_offline_alert(self.sensor)
        self._handle("OFFLINE")
        self.assertIn(self.sensor.id, self.cmd._last_status_seen)
        self.cmd._last_status_seen[self.sensor.id] = (
            self.sensor, timezone.now() - timedelta(seconds=OFFLINE_THRESHOLD_S + 60))
        self.cmd._check_offline_sensors()
        self.assertEqual(self._active_offline().count(), 1)
        self.assertEqual(self._active_offline().first().id, alert.id)

    def test_offline_status_without_alert_creates_one_after_threshold(self):
        self._handle("OFFLINE")
        self.assertEqual(self._active_offline().count(), 0)
        self.cmd._last_status_seen[self.sensor.id] = (
            self.sensor, timezone.now() - timedelta(seconds=OFFLINE_THRESHOLD_S + 60))
        self.cmd._check_offline_sensors()
        self.assertEqual(self._active_offline().count(), 1)
