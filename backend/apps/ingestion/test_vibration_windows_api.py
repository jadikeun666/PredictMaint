"""API /vibration-windows/ (ENGINEER, ADMIN; read-only)."""
from datetime import timedelta, timezone as dt_timezone
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APITestCase

from apps.registry.models import Bearing, Machine, Sensor

from .models import VibrationWindow

User = get_user_model()
URL = "/api/v1/vibration-windows/"


def make_user(username, role):
    return User.objects.create_user(username=username, password="x", role=role)


def iso(dt):
    return dt.astimezone(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class VibrationWindowApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = make_user("vw_admin", User.Role.ADMIN)
        cls.engineer = make_user("vw_eng", User.Role.ENGINEER)
        cls.technician = make_user("vw_tech", User.Role.TECHNICIAN)
        cls.viewer = make_user("vw_view", User.Role.VIEWER)
        m = Machine.objects.create(name="Vw Rig", location="L", machine_type="t", criticality="LOW")
        cls.b1 = Bearing.objects.create(machine=m, position_label="drive-end", bearing_model="X", status="ACTIVE")
        cls.b2 = Bearing.objects.create(machine=m, position_label="fan-end", bearing_model="X", status="ACTIVE")
        cls.s1 = Sensor.objects.create(bearing=cls.b1, sensor_type="ACCELEROMETER", axis="RADIAL",
                                       sample_rate_hz=12000, source="SIMULATED", mqtt_topic="factory/vw-rig/drive-end/vibration")
        cls.s2 = Sensor.objects.create(bearing=cls.b2, sensor_type="ACCELEROMETER", axis="RADIAL",
                                       sample_rate_hz=12000, source="SIMULATED", mqtt_topic="factory/vw-rig/fan-end/vibration")
        cls.now = timezone.now().replace(microsecond=0)

    def win(self, sensor=None, age_min=0, dataset="CWRU"):
        return VibrationWindow.objects.create(
            sensor=sensor or self.s1, captured_at=self.now - timedelta(minutes=age_min),
            duration_s=Decimal("1.000"), num_samples=12000, waveform_path=f"x{age_min}.npy",
            source_dataset=dataset,
        )

    def test_unauthenticated_401(self):
        self.assertEqual(self.client.get(URL).status_code, 401)

    def test_only_engineer_and_admin_read(self):
        w = self.win()
        for u in (self.engineer, self.admin):
            self.client.force_authenticate(u)
            self.assertEqual(self.client.get(URL).status_code, 200, u.role)
            r = self.client.get(f"{URL}{w.id}/")
            self.assertEqual(r.status_code, 200)
            for f in ("id", "sensor", "captured_at", "duration_s", "num_samples",
                      "waveform_path", "source_dataset", "ingested_at"):
                self.assertIn(f, r.json())
        for u in (self.technician, self.viewer):
            self.client.force_authenticate(u)
            self.assertEqual(self.client.get(URL).status_code, 403, u.role)
            self.assertEqual(self.client.get(f"{URL}{w.id}/").status_code, 403, u.role)

    def test_no_write_endpoints_405(self):
        w = self.win()
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.post(URL, {}, format="json").status_code, 405)
        for method in ("put", "patch", "delete"):
            self.assertEqual(getattr(self.client, method)(f"{URL}{w.id}/").status_code, 405, method)
        self.assertTrue(VibrationWindow.objects.filter(id=w.id).exists())

    def test_ordering_newest_first_and_cursor_pagination(self):
        old, new = self.win(age_min=60), self.win(age_min=1)
        self.client.force_authenticate(self.engineer)
        ids = [x["id"] for x in self.client.get(URL).json()["results"]]
        self.assertEqual(ids, [str(new.id), str(old.id)])
        for i in range(2, 55):
            self.win(age_min=100 + i)
        p1 = self.client.get(URL).json()
        self.assertEqual(len(p1["results"]), 50)
        p2 = self.client.get(p1["next"]).json()
        self.assertEqual(len(p2["results"]), 55 - 50)
        self.assertIsNone(p2["next"])
        self.assertFalse({x["id"] for x in p1["results"]} & {x["id"] for x in p2["results"]})

    def test_filters(self):
        a = self.win(self.s1, 10, "CWRU")
        b = self.win(self.s2, 20, "IMS")
        self.client.force_authenticate(self.engineer)

        def ids(**p):
            return {x["id"] for x in self.client.get(URL, p).json()["results"]}

        self.assertEqual(ids(sensor=str(self.s2.id)), {str(b.id)})
        self.assertEqual(ids(bearing=str(self.b1.id)), {str(a.id)})
        self.assertEqual(ids(source_dataset="IMS"), {str(b.id)})
        self.assertEqual(ids(source_dataset="CWRU,IMS"), {str(a.id), str(b.id)})
        self.assertEqual(ids(start=iso(self.now - timedelta(minutes=10))), {str(a.id)})   # inklusif
        self.assertEqual(ids(end=iso(self.now - timedelta(minutes=10))), {str(b.id)})     # eksklusif

    def test_invalid_params_400(self):
        self.client.force_authenticate(self.engineer)
        t = self.now
        for p in ({"sensor": "x"}, {"bearing": "x"}, {"source_dataset": "cwru"}, {"start": "kemarin"},
                  {"start": iso(t), "end": iso(t - timedelta(days=1))}):
            r = self.client.get(URL, p)
            self.assertEqual(r.status_code, 400, p)
            self.assertEqual(r.json()["code"], "invalid_query_param")
