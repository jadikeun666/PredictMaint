"""API alerts: list/retrieve/filter/paginasi + POST resolve (api.md)."""
import uuid
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APITestCase

from apps.registry.models import Bearing, Machine
from apps.scheduling.models import WorkOrder

from .models import Alert

User = get_user_model()
URL = "/api/v1/alerts/"


def make_user(username, role):
    return User.objects.create_user(username=username, password="x", role=role)


class AlertApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = make_user("al_admin", User.Role.ADMIN)
        cls.engineer = make_user("al_eng", User.Role.ENGINEER)
        cls.technician = make_user("al_tech", User.Role.TECHNICIAN)
        cls.viewer = make_user("al_view", User.Role.VIEWER)
        m = Machine.objects.create(name="Al Rig", location="L", machine_type="t", criticality="LOW")
        cls.b1 = Bearing.objects.create(machine=m, position_label="drive-end", bearing_model="X", status="ACTIVE")
        cls.b2 = Bearing.objects.create(machine=m, position_label="fan-end", bearing_model="X", status="ACTIVE")
        cls.now = timezone.now()

    def alert(self, bearing=None, alert_type="FAULT_DETECTED", severity="CRITICAL", age_min=0, resolved=False):
        t = self.now - timedelta(minutes=age_min)
        return Alert.objects.create(
            bearing=bearing, alert_type=alert_type, severity=severity,
            triggered_at=t, last_seen_at=t,
            resolved_at=t if resolved else None,
        )

    # ---- akses
    def test_unauthenticated_401(self):
        a = self.alert(self.b1)
        self.assertEqual(self.client.get(URL).status_code, 401)
        self.assertEqual(self.client.get(f"{URL}{a.id}/").status_code, 401)
        self.assertEqual(self.client.post(f"{URL}{a.id}/resolve/").status_code, 401)

    def test_all_roles_read(self):
        a = self.alert(self.b1)
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            r = self.client.get(URL)
            self.assertEqual(r.status_code, 200, u.role)
            self.assertIn("results", r.json())
            r = self.client.get(f"{URL}{a.id}/")
            self.assertEqual(r.status_code, 200, u.role)
            body = r.json()
            for f in ("id", "bearing", "spare_part", "alert_type", "severity", "related_prediction",
                      "triggered_at", "last_seen_at", "resolved_at", "resolved_by"):
                self.assertIn(f, body)

    def test_no_manual_write_405(self):
        a = self.alert(self.b1)
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.post(URL, {}, format="json").status_code, 405)
        for method in ("put", "patch", "delete"):
            self.assertEqual(getattr(self.client, method)(f"{URL}{a.id}/").status_code, 405, method)

    # ---- list: urutan, paginasi, filter
    def test_ordering_newest_first(self):
        old = self.alert(self.b1, age_min=60)
        new = self.alert(self.b1, age_min=1)
        mid = self.alert(self.b1, age_min=30)
        self.client.force_authenticate(self.viewer)
        ids = [x["id"] for x in self.client.get(URL).json()["results"]]
        self.assertEqual(ids, [str(new.id), str(mid.id), str(old.id)])

    def test_cursor_pagination(self):
        for i in range(55):
            self.alert(self.b1, age_min=i)
        self.client.force_authenticate(self.viewer)
        p1 = self.client.get(URL).json()
        self.assertEqual(len(p1["results"]), 50)
        self.assertIsNotNone(p1["next"])
        p2 = self.client.get(p1["next"]).json()
        self.assertEqual(len(p2["results"]), 5)
        self.assertIsNone(p2["next"])
        self.assertFalse({x["id"] for x in p1["results"]} & {x["id"] for x in p2["results"]})

    def test_filters(self):
        a1 = self.alert(self.b1, "FAULT_DETECTED", "CRITICAL")
        a2 = self.alert(self.b2, "RUL_THRESHOLD", "HIGH")
        a3 = self.alert(None, "LOW_STOCK", "WARNING", resolved=True)
        self.client.force_authenticate(self.viewer)

        def ids(**p):
            return {x["id"] for x in self.client.get(URL, p).json()["results"]}

        self.assertEqual(ids(bearing=str(self.b1.id)), {str(a1.id)})
        self.assertEqual(ids(alert_type="LOW_STOCK"), {str(a3.id)})
        self.assertEqual(ids(alert_type="LOW_STOCK,RUL_THRESHOLD"), {str(a2.id), str(a3.id)})
        self.assertEqual(ids(severity="HIGH"), {str(a2.id)})
        self.assertEqual(ids(resolved="true"), {str(a3.id)})
        self.assertEqual(ids(resolved="false"), {str(a1.id), str(a2.id)})
        self.assertEqual(ids(severity="CRITICAL", resolved="false"), {str(a1.id)})

    def test_invalid_filters_400_rfc7807(self):
        self.client.force_authenticate(self.viewer)
        for params in ({"alert_type": "NOPE"}, {"severity": "critical"}, {"resolved": "maybe"}, {"bearing": "x"}):
            r = self.client.get(URL, params)
            self.assertEqual(r.status_code, 400, params)
            body = r.json()
            for k in ("type", "title", "status", "detail"):
                self.assertIn(k, body)

    # ---- resolve
    def test_resolve_engineer_and_admin(self):
        for u in (self.engineer, self.admin):
            a = self.alert(self.b1)
            self.client.force_authenticate(u)
            r = self.client.post(f"{URL}{a.id}/resolve/")
            self.assertEqual(r.status_code, 200, u.role)
            body = r.json()
            self.assertIsNotNone(body["resolved_at"])
            self.assertEqual(body["resolved_by"], str(u.id))
            a.refresh_from_db()
            self.assertEqual(a.resolved_by_id, u.id)
            self.assertIsNotNone(a.resolved_at)

    def test_resolve_forbidden_for_technician_and_viewer(self):
        a = self.alert(self.b1)
        for u in (self.technician, self.viewer):
            self.client.force_authenticate(u)
            self.assertEqual(self.client.post(f"{URL}{a.id}/resolve/").status_code, 403, u.role)
        a.refresh_from_db()
        self.assertIsNone(a.resolved_at)
        self.assertIsNone(a.resolved_by_id)

    def test_resolve_already_resolved_409_and_unchanged(self):
        a = self.alert(self.b1)
        self.client.force_authenticate(self.engineer)
        self.assertEqual(self.client.post(f"{URL}{a.id}/resolve/").status_code, 200)
        a.refresh_from_db()
        first_at, first_by = a.resolved_at, a.resolved_by_id
        self.client.force_authenticate(self.admin)
        r = self.client.post(f"{URL}{a.id}/resolve/")
        self.assertEqual(r.status_code, 409)
        body = r.json()
        for k in ("type", "title", "status", "detail"):
            self.assertIn(k, body)
        a.refresh_from_db()
        self.assertEqual(a.resolved_at, first_at)
        self.assertEqual(a.resolved_by_id, first_by)

    def test_resolve_system_resolved_alert_409(self):
        a = self.alert(self.b1, resolved=True)  # resolved_by null (system)
        self.client.force_authenticate(self.engineer)
        self.assertEqual(self.client.post(f"{URL}{a.id}/resolve/").status_code, 409)
        a.refresh_from_db()
        self.assertIsNone(a.resolved_by_id)

    def test_resolve_unknown_id_404(self):
        self.client.force_authenticate(self.engineer)
        self.assertEqual(self.client.post(f"{URL}{uuid.uuid4()}/resolve/").status_code, 404)

    def test_resolve_does_not_cascade_to_work_order(self):
        a = self.alert(self.b1)
        wo = WorkOrder.objects.create(
            bearing=self.b1, alert=a, status="SCHEDULED", priority="URGENT",
            scheduled_start=self.now, scheduled_end=self.now + timedelta(hours=4),
        )
        self.client.force_authenticate(self.engineer)
        self.assertEqual(self.client.post(f"{URL}{a.id}/resolve/").status_code, 200)
        wo.refresh_from_db()
        self.assertEqual(wo.status, "SCHEDULED")
        self.assertEqual(wo.alert_id, a.id)

    def test_resolve_alert_without_bearing(self):
        a = self.alert(None, "LOW_STOCK", "WARNING")
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.post(f"{URL}{a.id}/resolve/").status_code, 200)
