"""Matriks role + DELETE aman untuk /machines/ (Giliran 1)."""
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from .models import Bearing, Machine, Sensor

User = get_user_model()
URL = "/api/v1/machines/"


def make_user(username, role):
    try:
        u = User.objects.create_user(username=username, password="x", role=role)
    except Exception:
        u = User.objects.create(username=username, role=role)
    return u


class MachineApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = make_user("t_admin", User.Role.ADMIN)
        cls.engineer = make_user("t_eng", User.Role.ENGINEER)
        cls.technician = make_user("t_tech", User.Role.TECHNICIAN)
        cls.viewer = make_user("t_view", User.Role.VIEWER)
        cls.machine = Machine.objects.create(
            name="Test Rig", location="Lab", machine_type="rig", criticality="LOW"
        )

    payload = {
        "name": "New Rig",
        "location": "Plant A",
        "machine_type": "pump",
        "criticality": "HIGH",
    }

    def test_unauthenticated_401(self):
        self.assertEqual(self.client.get(URL).status_code, 401)

    def test_all_roles_can_read(self):
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            r = self.client.get(URL)
            self.assertEqual(r.status_code, 200, u.role)
            self.assertIn("results", r.json())
            r = self.client.get(f"{URL}{self.machine.id}/")
            self.assertEqual(r.status_code, 200, u.role)

    def test_non_admin_cannot_write(self):
        for u in (self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            self.assertEqual(self.client.post(URL, self.payload, format="json").status_code, 403, u.role)
            self.assertEqual(
                self.client.patch(f"{URL}{self.machine.id}/", {"name": "x"}, format="json").status_code, 403, u.role
            )
            self.assertEqual(self.client.delete(f"{URL}{self.machine.id}/").status_code, 403, u.role)
        self.assertTrue(Machine.objects.filter(id=self.machine.id, name="Test Rig").exists())

    def test_admin_crud(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post(URL, self.payload, format="json")
        self.assertEqual(r.status_code, 201, r.content)
        mid = r.json()["id"]
        r = self.client.patch(f"{URL}{mid}/", {"location": "Plant B"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["location"], "Plant B")
        self.assertEqual(self.client.delete(f"{URL}{mid}/").status_code, 204)
        self.assertFalse(Machine.objects.filter(id=mid).exists())

    def test_invalid_criticality_400(self):
        self.client.force_authenticate(self.admin)
        bad = dict(self.payload, criticality="EXTREME")
        self.assertEqual(self.client.post(URL, bad, format="json").status_code, 400)

    def test_delete_protected_returns_409_rfc7807(self):
        Bearing.objects.create(
            machine=self.machine,
            position_label="drive-end",
            bearing_model="TEST",
            status=Bearing.Status.ACTIVE,
        )
        self.client.force_authenticate(self.admin)
        r = self.client.delete(f"{URL}{self.machine.id}/")
        self.assertEqual(r.status_code, 409)
        body = r.json()
        for k in ("type", "title", "status", "detail"):
            self.assertIn(k, body)
        self.assertEqual(body["status"], 409)
        self.assertIn("registry.Bearing", body["detail"])
        self.assertTrue(Machine.objects.filter(id=self.machine.id).exists())


BURL = "/api/v1/bearings/"


class BearingApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = make_user("b_admin", User.Role.ADMIN)
        cls.engineer = make_user("b_eng", User.Role.ENGINEER)
        cls.technician = make_user("b_tech", User.Role.TECHNICIAN)
        cls.viewer = make_user("b_view", User.Role.VIEWER)
        cls.machine = Machine.objects.create(
            name="Rig B", location="Lab", machine_type="rig", criticality="LOW"
        )
        cls.bearing = Bearing.objects.create(
            machine=cls.machine, position_label="drive-end",
            bearing_model="SKF X", status="ACTIVE",
        )

    def payload(self, **kw):
        base = {
            "machine": str(self.machine.id),
            "position_label": "fan-end",
            "bearing_model": "SKF Y",
            "status": "ACTIVE",
        }
        base.update(kw)
        return base

    def test_unauthenticated_401(self):
        self.assertEqual(self.client.get(BURL).status_code, 401)

    def test_all_roles_read_and_list_paginates(self):
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            r = self.client.get(BURL)
            self.assertEqual(r.status_code, 200, u.role)
            self.assertIn("results", r.json())
            self.assertEqual(self.client.get(f"{BURL}{self.bearing.id}/").status_code, 200)

    def test_non_admin_cannot_write(self):
        for u in (self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            self.assertEqual(self.client.post(BURL, self.payload(), format="json").status_code, 403, u.role)
            self.assertEqual(self.client.patch(f"{BURL}{self.bearing.id}/", {"bearing_model": "z"}, format="json").status_code, 403)
            self.assertEqual(self.client.delete(f"{BURL}{self.bearing.id}/").status_code, 403)

    def test_create_without_geometry_then_patch_geometry(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post(BURL, self.payload(), format="json")
        self.assertEqual(r.status_code, 201, r.content)
        body = r.json()
        for f in ("ball_diameter_mm", "pitch_diameter_mm", "contact_angle_deg", "num_balls"):
            self.assertIsNone(body[f])
        bid = body["id"]
        r = self.client.patch(
            f"{BURL}{bid}/",
            {"ball_diameter_mm": "7.940", "pitch_diameter_mm": "39.040",
             "contact_angle_deg": "0.00", "num_balls": 9},
            format="json",
        )
        self.assertEqual(r.status_code, 200, r.content)
        b = Bearing.objects.get(id=bid)
        self.assertEqual(float(b.ball_diameter_mm), 7.94)
        self.assertEqual(float(b.pitch_diameter_mm), 39.04)
        self.assertEqual(b.num_balls, 9)

    def test_patch_partial_geometry_one_field_at_a_time(self):
        self.client.force_authenticate(self.admin)
        r = self.client.patch(f"{BURL}{self.bearing.id}/", {"num_balls": 9}, format="json")
        self.assertEqual(r.status_code, 200, r.content)
        r = self.client.patch(f"{BURL}{self.bearing.id}/", {"pitch_diameter_mm": "39.040"}, format="json")
        self.assertEqual(r.status_code, 200, r.content)

    def test_geometry_sanity_d_less_than_D(self):
        self.client.force_authenticate(self.admin)
        r = self.client.patch(
            f"{BURL}{self.bearing.id}/",
            {"ball_diameter_mm": "50.000", "pitch_diameter_mm": "39.040"}, format="json",
        )
        self.assertEqual(r.status_code, 400)
        r = self.client.patch(f"{BURL}{self.bearing.id}/", {"num_balls": 0}, format="json")
        self.assertEqual(r.status_code, 400)

    def test_machine_fk_must_exist(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post(BURL, self.payload(machine="00000000-0000-0000-0000-000000000000"), format="json")
        self.assertEqual(r.status_code, 400)

    def test_position_label_validation_and_uniqueness(self):
        self.client.force_authenticate(self.admin)
        for bad in ("a/b", "has space", "x+y", "x#"):
            r = self.client.post(BURL, self.payload(position_label=bad), format="json")
            self.assertEqual(r.status_code, 400, bad)
        r = self.client.post(BURL, self.payload(position_label="drive-end"), format="json")
        self.assertEqual(r.status_code, 400)

    def test_label_and_machine_locked_when_sensor_exists(self):
        Sensor.objects.create(
            bearing=self.bearing, sensor_type="ACCELEROMETER", axis="RADIAL",
            sample_rate_hz=12000, source="SIMULATED",
            mqtt_topic="factory/rig-b/drive-end/vibration",
        )
        self.client.force_authenticate(self.admin)
        r = self.client.patch(f"{BURL}{self.bearing.id}/", {"position_label": "other"}, format="json")
        self.assertEqual(r.status_code, 400)
        r = self.client.patch(f"{BURL}{self.bearing.id}/", {"bearing_model": "SKF Z"}, format="json")
        self.assertEqual(r.status_code, 200)

    def test_machine_rename_blocked_when_sensor_exists(self):
        Sensor.objects.create(
            bearing=self.bearing, sensor_type="ACCELEROMETER", axis="RADIAL",
            sample_rate_hz=12000, source="SIMULATED",
            mqtt_topic="factory/rig-b/drive-end/vibration",
        )
        self.client.force_authenticate(self.admin)
        r = self.client.patch(f"{URL}{self.machine.id}/", {"name": "Rig Baru"}, format="json")
        self.assertEqual(r.status_code, 400)
        r = self.client.patch(f"{URL}{self.machine.id}/", {"location": "Plant Z"}, format="json")
        self.assertEqual(r.status_code, 200)
        r = self.client.patch(f"{URL}{self.machine.id}/", {"name": "RIG  B"}, format="json")
        self.assertEqual(r.status_code, 200)  # slug sama -> aman

    def test_delete_with_sensor_returns_409(self):
        Sensor.objects.create(
            bearing=self.bearing, sensor_type="ACCELEROMETER", axis="RADIAL",
            sample_rate_hz=12000, source="SIMULATED",
            mqtt_topic="factory/rig-b/drive-end/vibration",
        )
        self.client.force_authenticate(self.admin)
        r = self.client.delete(f"{BURL}{self.bearing.id}/")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["status"], 409)
        self.assertIn("registry.Sensor", r.json()["detail"])
        self.assertTrue(Bearing.objects.filter(id=self.bearing.id).exists())

    def test_delete_without_dependencies_204(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post(BURL, self.payload(), format="json")
        bid = r.json()["id"]
        self.assertEqual(self.client.delete(f"{BURL}{bid}/").status_code, 204)


from django.utils import timezone  # noqa: E402

from apps.ingestion.models import VibrationWindow  # noqa: E402

SURL = "/api/v1/sensors/"


class SensorApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = make_user("s_admin", User.Role.ADMIN)
        cls.engineer = make_user("s_eng", User.Role.ENGINEER)
        cls.technician = make_user("s_tech", User.Role.TECHNICIAN)
        cls.viewer = make_user("s_view", User.Role.VIEWER)
        cls.machine = Machine.objects.create(
            name="Rig S", location="Lab", machine_type="rig", criticality="LOW"
        )
        cls.bearing = Bearing.objects.create(
            machine=cls.machine, position_label="drive-end",
            bearing_model="SKF S", status="ACTIVE",
        )
        cls.bearing2 = Bearing.objects.create(
            machine=cls.machine, position_label="fan-end",
            bearing_model="SKF S2", status="ACTIVE",
        )

    def payload(self, bearing=None, **kw):
        base = {
            "bearing": str((bearing or self.bearing).id),
            "sensor_type": "ACCELEROMETER",
            "axis": "RADIAL",
            "sample_rate_hz": 12000,
            "source": "SIMULATED",
        }
        base.update(kw)
        return base

    def test_unauthenticated_401(self):
        self.assertEqual(self.client.get(SURL).status_code, 401)

    def test_all_roles_read(self):
        s = Sensor.objects.create(
            bearing=self.bearing, sensor_type="ACCELEROMETER", axis="RADIAL",
            sample_rate_hz=12000, source="SIMULATED",
            mqtt_topic="factory/rig-s/drive-end/vibration",
        )
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            r = self.client.get(SURL)
            self.assertEqual(r.status_code, 200, u.role)
            self.assertIn("results", r.json())
            self.assertEqual(self.client.get(f"{SURL}{s.id}/").status_code, 200)

    def test_non_admin_cannot_write(self):
        s = Sensor.objects.create(
            bearing=self.bearing, sensor_type="ACCELEROMETER", axis="RADIAL",
            sample_rate_hz=12000, source="SIMULATED",
            mqtt_topic="factory/rig-s/drive-end/vibration",
        )
        for u in (self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            self.assertEqual(self.client.post(SURL, self.payload(self.bearing2), format="json").status_code, 403, u.role)
            self.assertEqual(self.client.patch(f"{SURL}{s.id}/", {"sample_rate_hz": 48000}, format="json").status_code, 403)
            self.assertEqual(self.client.delete(f"{SURL}{s.id}/").status_code, 403)

    def test_create_generates_topic_and_ignores_client_topic(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post(SURL, self.payload(mqtt_topic="factory/evil/topic/vibration"), format="json")
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.json()["mqtt_topic"], "factory/rig-s/drive-end/vibration")

    def test_second_sensor_same_bearing_rejected_400(self):
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.post(SURL, self.payload(), format="json").status_code, 201)
        r = self.client.post(SURL, self.payload(axis="AXIAL"), format="json")
        self.assertEqual(r.status_code, 400)

    def test_bearing_fk_must_exist(self):
        self.client.force_authenticate(self.admin)
        p = self.payload()
        p["bearing"] = "00000000-0000-0000-0000-000000000000"
        r = self.client.post(SURL, p, format="json")
        self.assertEqual(r.status_code, 400)

    def test_invalid_enum_and_sample_rate_400(self):
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.post(SURL, self.payload(self.bearing2, axis="DIAGONAL"), format="json").status_code, 400)
        self.assertEqual(self.client.post(SURL, self.payload(self.bearing2, sample_rate_hz=0), format="json").status_code, 400)

    def test_patch_keeps_topic_and_blocks_bearing_move(self):
        self.client.force_authenticate(self.admin)
        sid = self.client.post(SURL, self.payload(), format="json").json()["id"]
        r = self.client.patch(f"{SURL}{sid}/", {"sample_rate_hz": 48000}, format="json")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["sample_rate_hz"], 48000)
        self.assertEqual(r.json()["mqtt_topic"], "factory/rig-s/drive-end/vibration")
        r = self.client.patch(f"{SURL}{sid}/", {"bearing": str(self.bearing2.id)}, format="json")
        self.assertEqual(r.status_code, 400)
        r = self.client.patch(f"{SURL}{sid}/", {"mqtt_topic": "factory/x/y/vibration"}, format="json")
        self.assertEqual(r.json()["mqtt_topic"], "factory/rig-s/drive-end/vibration")

    def test_delete_without_dependencies_204(self):
        self.client.force_authenticate(self.admin)
        sid = self.client.post(SURL, self.payload(), format="json").json()["id"]
        self.assertEqual(self.client.delete(f"{SURL}{sid}/").status_code, 204)

    def test_delete_with_vibration_window_409(self):
        s = Sensor.objects.create(
            bearing=self.bearing, sensor_type="ACCELEROMETER", axis="RADIAL",
            sample_rate_hz=12000, source="SIMULATED",
            mqtt_topic="factory/rig-s/drive-end/vibration",
        )
        VibrationWindow.objects.create(
            sensor=s, captured_at=timezone.now(), duration_s=1, num_samples=12000,
            waveform_path="x.npy", source_dataset="CWRU", ingested_at=timezone.now(),
        )
        self.client.force_authenticate(self.admin)
        r = self.client.delete(f"{SURL}{s.id}/")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["status"], 409)
        self.assertIn("ingestion.VibrationWindow", r.json()["detail"])
        self.assertTrue(Sensor.objects.filter(id=s.id).exists())
