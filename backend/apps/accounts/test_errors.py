"""Exception handler RFC7807 project-wide (config/exception_handler.py)."""
import uuid
from unittest import mock

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from apps.registry.models import Bearing, Machine, Sensor
from apps.registry.views import MachineViewSet

User = get_user_model()
PROBLEM = "application/problem+json"
REQUIRED = ("type", "title", "status", "detail", "code")


def make_user(username, role):
    return User.objects.create_user(username=username, password="x", role=role)


class ProblemHandlerTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = make_user("eh_admin", User.Role.ADMIN)
        cls.viewer = make_user("eh_view", User.Role.VIEWER)
        cls.machine = Machine.objects.create(name="Eh Rig", location="L", machine_type="t", criticality="LOW")

    def assertProblem(self, r, status, code, title=None):
        self.assertEqual(r.status_code, status, r.content)
        body = r.json()
        for k in REQUIRED:
            self.assertIn(k, body)
        self.assertEqual(body["type"], "about:blank")
        self.assertEqual(body["status"], status)
        self.assertEqual(body["code"], code)
        if title:
            self.assertEqual(body["title"], title)
        self.assertTrue(r["Content-Type"].startswith(PROBLEM), r["Content-Type"])
        return body

    # ---- autentikasi
    def test_401_no_token(self):
        r = self.client.get("/api/v1/machines/")
        body = self.assertProblem(r, 401, "not_authenticated", "Unauthorized")
        self.assertNotIn("errors", body)
        self.assertIn("Bearer", r["WWW-Authenticate"])  # header asli dipertahankan

    def test_401_invalid_token_drops_extra_simplejwt_fields(self):
        self.client.credentials(HTTP_AUTHORIZATION="Bearer abc")
        body = self.assertProblem(self.client.get("/api/v1/machines/"), 401, "token_not_valid")
        self.assertNotIn("messages", body)

    def test_401_login_wrong_password(self):
        r = self.client.post("/api/v1/auth/token/", {"username": "eh_admin", "password": "salah"}, format="json")
        self.assertProblem(r, 401, "no_active_account")

    def test_400_login_empty_is_validation_error(self):
        r = self.client.post("/api/v1/auth/token/", {}, format="json")
        body = self.assertProblem(r, 400, "validation_error", "Bad Request")
        self.assertEqual(set(body["errors"]), {"username", "password"})
        self.assertIsInstance(body["errors"]["username"], list)
        self.assertIn("username", body["detail"])

    # ---- 403 / 404 / 405 / 406 / 415 / parse error
    def test_403_permission_denied(self):
        self.client.force_authenticate(self.viewer)
        r = self.client.post("/api/v1/machines/", {}, format="json")
        self.assertProblem(r, 403, "permission_denied", "Forbidden")

    def test_404_missing_object(self):
        self.client.force_authenticate(self.admin)
        self.assertProblem(self.client.get(f"/api/v1/machines/{uuid.uuid4()}/"), 404, "not_found", "Not Found")

    def test_404_invalid_uuid_in_path(self):
        self.client.force_authenticate(self.admin)
        self.assertProblem(self.client.get("/api/v1/machines/bukan-uuid/"), 404, "not_found")

    def test_400_parse_error(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post("/api/v1/machines/", data="{bad", content_type="application/json")
        body = self.assertProblem(r, 400, "parse_error")
        self.assertIn("JSON parse error", body["detail"])

    def test_415_unsupported_media_type(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post("/api/v1/machines/", data="x", content_type="text/plain")
        self.assertProblem(r, 415, "unsupported_media_type")

    def test_405_method_not_allowed(self):
        self.client.force_authenticate(self.admin)
        self.assertProblem(self.client.post("/api/v1/alerts/", {}, format="json"), 405, "method_not_allowed")

    def test_406_not_acceptable(self):
        self.client.force_authenticate(self.admin)
        r = self.client.get("/api/v1/machines/", HTTP_ACCEPT="application/xml")
        self.assertEqual(r.status_code, 406)
        self.assertEqual(r.json()["code"], "not_acceptable")

    # ---- validasi serializer
    def test_400_serializer_validation_errors_per_field(self):
        self.client.force_authenticate(self.admin)
        body = self.assertProblem(self.client.post("/api/v1/machines/", {}, format="json"), 400, "validation_error")
        self.assertEqual(set(body["errors"]), {"name", "machine_type", "criticality"})
        for msgs in body["errors"].values():
            self.assertIsInstance(msgs, list)
            self.assertTrue(all(isinstance(m, str) for m in msgs))
        self.assertTrue(body["detail"].startswith("Data tidak valid:"))

    def test_400_cross_field_validation_keeps_field_key(self):
        b = Bearing.objects.create(machine=self.machine, position_label="de", bearing_model="X", status="ACTIVE")
        self.client.force_authenticate(self.admin)
        r = self.client.patch(
            f"/api/v1/bearings/{b.id}/",
            {"ball_diameter_mm": "50.000", "pitch_diameter_mm": "39.040"}, format="json",
        )
        body = self.assertProblem(r, 400, "validation_error")
        self.assertIn("ball_diameter_mm", body["errors"])

    def test_400_object_level_error_goes_to_non_field_errors(self):
        b = Bearing.objects.create(machine=self.machine, position_label="de", bearing_model="X", status="ACTIVE")
        Sensor.objects.create(
            bearing=b, sensor_type="ACCELEROMETER", axis="RADIAL", sample_rate_hz=12000,
            source="SIMULATED", mqtt_topic="factory/eh-rig/de/vibration",
        )
        self.client.force_authenticate(self.admin)
        r = self.client.patch(f"/api/v1/bearings/{b.id}/", {"position_label": "lain"}, format="json")
        body = self.assertProblem(r, 400, "validation_error")
        self.assertIn("non_field_errors", body["errors"])

    # ---- 404 URL tak dikenal (handler404; test runner memaksa DEBUG=False)
    def test_unknown_api_url_returns_json_404(self):
        r = self.client.get("/api/v1/tidak-ada/")
        self.assertProblem(r, 404, "not_found", "Not Found")

    def test_non_api_404_keeps_django_default_page(self):
        r = self.client.get("/tidak-ada-lagi/")
        self.assertEqual(r.status_code, 404)
        self.assertTrue(r["Content-Type"].startswith("text/html"))

    # ---- 500
    def test_unhandled_exception_returns_generic_500_and_logs(self):
        self.client.force_authenticate(self.admin)
        with mock.patch.object(MachineViewSet, "list", side_effect=RuntimeError("secret-internal-detail")):
            with self.assertLogs("django.request", level="ERROR") as logs:
                r = self.client.get("/api/v1/machines/")
        body = self.assertProblem(r, 500, "server_error", "Internal Server Error")
        self.assertNotIn("secret-internal-detail", r.content.decode())
        self.assertNotIn("Traceback", r.content.decode())
        self.assertIn("secret-internal-detail", "\n".join(logs.output))  # tercatat di log, bukan di klien

    # ---- Browsable API tidak dirusak
    def test_html_clients_do_not_get_problem_content_type(self):
        r = self.client.get("/api/v1/machines/", HTTP_ACCEPT="text/html")
        self.assertEqual(r.status_code, 401)
        self.assertTrue(r["Content-Type"].startswith("text/html"), r["Content-Type"])
