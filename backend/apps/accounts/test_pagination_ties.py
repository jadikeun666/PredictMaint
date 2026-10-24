"""
Regresi: cursor pagination harus stabil saat banyak baris punya timestamp SAMA
(ditemukan pada data nyata vibration_windows: 4040 baris, id tidak unik antar halaman).
"""
import datetime as dt
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APITestCase

from apps.alerting.models import Alert
from apps.ingestion.models import VibrationWindow
from apps.ml_inference.models import Prediction
from apps.registry.models import Bearing, Machine, Sensor

User = get_user_model()
N = 120  # > 2 halaman (PAGE_SIZE 50) dengan seluruh baris bertimestamp identik


class CursorTieTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = User.objects.create_user(username="tie_admin", password="x", role=User.Role.ADMIN)
        cls.engineer = User.objects.create_user(username="tie_eng", password="x", role=User.Role.ENGINEER)
        cls.machine = Machine.objects.create(name="Tie Rig", location="L", machine_type="t", criticality="LOW")
        cls.bearing = Bearing.objects.create(machine=cls.machine, position_label="de", bearing_model="X", status="ACTIVE")
        cls.sensor = Sensor.objects.create(
            bearing=cls.bearing, sensor_type="ACCELEROMETER", axis="RADIAL", sample_rate_hz=12000,
            source="SIMULATED", mqtt_topic="factory/tie-rig/de/vibration",
        )
        cls.t = timezone.now().replace(microsecond=0) - dt.timedelta(hours=1)

    def walk(self, url, user):
        self.client.force_authenticate(user)
        ids, nxt, pages = [], url, 0
        while nxt:
            d = self.client.get(nxt).json()
            ids += [x["id"] for x in d["results"]]
            nxt = d["next"]
            pages += 1
        return ids, pages

    def assertStable(self, url, user):
        ids, pages = self.walk(url, user)
        self.assertGreaterEqual(pages, 3)
        self.assertEqual(len(ids), N)
        self.assertEqual(len(set(ids)), N, "ada duplikat/terlewat antar halaman")

    def test_vibration_windows_identical_captured_at(self):
        VibrationWindow.objects.bulk_create([
            VibrationWindow(sensor=self.sensor, captured_at=self.t, duration_s=Decimal("1.000"),
                            num_samples=12000, waveform_path=f"w{i}.npy", source_dataset="CWRU")
            for i in range(N)
        ])
        self.assertStable("/api/v1/vibration-windows/", self.engineer)

    def test_predictions_identical_predicted_at(self):
        Prediction.objects.bulk_create([
            Prediction(bearing=self.bearing, prediction_type="CLASSIFICATION", fault_class="OUTER_RACE",
                       classification_confidence=Decimal("0.900"), model_version="tie-v1", predicted_at=self.t)
            for _ in range(N)
        ])
        self.assertStable("/api/v1/predictions/", self.engineer)

    def test_alerts_identical_triggered_at(self):
        Alert.objects.bulk_create([
            Alert(bearing=self.bearing, alert_type="FAULT_DETECTED", severity="HIGH",
                  triggered_at=self.t, last_seen_at=self.t)
            for _ in range(N)
        ])
        self.assertStable("/api/v1/alerts/", self.engineer)

    def test_machines_identical_created_at(self):
        for i in range(N):
            Machine.objects.create(name=f"Tie M{i}", location="L", machine_type="t", criticality="LOW")
        Machine.objects.update(created_at=self.t)
        ids, pages = self.walk("/api/v1/machines/", self.admin)
        total = Machine.objects.count()
        self.assertGreaterEqual(pages, 3)
        self.assertEqual(len(ids), total)
        self.assertEqual(len(set(ids)), total)
