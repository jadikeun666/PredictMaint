"""GET /gantt/ — filter, konflik (gantt.md §4), rentang, role."""
import uuid
from datetime import timedelta, timezone as dt_timezone

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APITestCase

from apps.registry.models import Bearing, Machine

from .models import WorkOrder

User = get_user_model()
URL = "/api/v1/gantt/"


def make_user(username, role):
    return User.objects.create_user(username=username, password="x", role=role)


def iso(dt):
    return dt.astimezone(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class GanttApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = make_user("g_admin", User.Role.ADMIN)
        cls.engineer = make_user("g_eng", User.Role.ENGINEER)
        cls.technician = make_user("g_tech", User.Role.TECHNICIAN)
        cls.viewer = make_user("g_view", User.Role.VIEWER)
        cls.m1 = Machine.objects.create(name="Gantt M1", location="L", machine_type="t", criticality="LOW")
        cls.m2 = Machine.objects.create(name="Gantt M2", location="L", machine_type="t", criticality="LOW")
        cls.m3 = Machine.objects.create(name="Gantt M3 Kosong", location="L", machine_type="t", criticality="LOW")
        cls.b1a = Bearing.objects.create(machine=cls.m1, position_label="drive-end", bearing_model="X", status="ACTIVE")
        cls.b1b = Bearing.objects.create(machine=cls.m1, position_label="fan-end", bearing_model="X", status="ACTIVE")
        cls.b2 = Bearing.objects.create(machine=cls.m2, position_label="drive-end", bearing_model="X", status="ACTIVE")
        cls.base = timezone.now().replace(microsecond=0) + timedelta(days=1)

    # ---- helpers
    def wo(self, bearing, start_h, dur_h, status="SCHEDULED", priority="HIGH", tech=None):
        s = self.base + timedelta(hours=start_h)
        return WorkOrder.objects.create(
            bearing=bearing, status=status, priority=priority,
            scheduled_start=s, scheduled_end=s + timedelta(hours=dur_h),
            assigned_technician=tech,
        )

    def get(self, **params):
        params.setdefault("start", iso(self.base - timedelta(days=1)))
        params.setdefault("end", iso(self.base + timedelta(days=2)))
        self.client.force_authenticate(self.admin)
        return self.client.get(URL, params)

    @staticmethod
    def bars(resp):
        """JSON mengembalikan id sebagai string; normalisasi ke UUID agar sebanding dengan wo.id."""
        out = {}
        for m in resp.json()["machines"]:
            for b in m["work_orders"]:
                b = dict(b)
                b["conflicts_with"] = [uuid.UUID(x) for x in b["conflicts_with"]]
                out[uuid.UUID(b["id"])] = b
        return out

    # ---- akses
    def test_unauthenticated_401(self):
        self.assertEqual(self.client.get(URL).status_code, 401)

    def test_all_roles_200_and_shape(self):
        self.wo(self.b1a, 0, 2)
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            r = self.client.get(URL, {"start": iso(self.base - timedelta(days=1)), "end": iso(self.base + timedelta(days=2))})
            self.assertEqual(r.status_code, 200, u.role)
            body = r.json()
            self.assertIn("range", body)
            self.assertIn("machines", body)

    def test_write_methods_405(self):
        self.client.force_authenticate(self.admin)
        for method in ("post", "put", "patch", "delete"):
            self.assertEqual(getattr(self.client, method)(URL).status_code, 405, method)

    # ---- sumber data & rentang
    def test_only_scheduled_and_in_progress(self):
        keep1 = self.wo(self.b1a, 0, 2, status="SCHEDULED")
        keep2 = self.wo(self.b2, 0, 2, status="IN_PROGRESS")
        for st in ("COMPLETED", "CANCELLED"):
            self.wo(self.b1a, 10, 2, status=st)
        WorkOrder.objects.create(bearing=self.b1a, status="DRAFT", priority="HIGH")  # tanpa jadwal
        self.wo(self.b1a, 20, 2, status="DRAFT")  # DRAFT dengan saran jadwal
        self.assertEqual(set(self.bars(self.get())), {keep1.id, keep2.id})

    def test_default_range_is_today_plus_14_days(self):
        now = timezone.now()
        inside = WorkOrder.objects.create(
            bearing=self.b1a, status="SCHEDULED", priority="LOW",
            scheduled_start=now + timedelta(days=2), scheduled_end=now + timedelta(days=2, hours=4))
        past = WorkOrder.objects.create(
            bearing=self.b1a, status="SCHEDULED", priority="LOW",
            scheduled_start=now - timedelta(days=3), scheduled_end=now - timedelta(days=3) + timedelta(hours=4))
        far = WorkOrder.objects.create(
            bearing=self.b1a, status="SCHEDULED", priority="LOW",
            scheduled_start=now + timedelta(days=20), scheduled_end=now + timedelta(days=20, hours=4))
        self.client.force_authenticate(self.admin)
        r = self.client.get(URL)  # tanpa parameter
        self.assertEqual(r.status_code, 200)
        ids = set(self.bars(r))
        self.assertIn(inside.id, ids)
        self.assertNotIn(past.id, ids)
        self.assertNotIn(far.id, ids)

    def test_explicit_range_includes_past(self):
        now = timezone.now()
        past = WorkOrder.objects.create(
            bearing=self.b1a, status="SCHEDULED", priority="LOW",
            scheduled_start=now - timedelta(days=3), scheduled_end=now - timedelta(days=3) + timedelta(hours=4))
        r = self.get(start=iso(now - timedelta(days=5)), end=iso(now - timedelta(days=1)))
        self.assertIn(past.id, self.bars(r))

    # ---- konflik (gantt.md §4)
    def test_conflict_same_machine_overlap(self):
        a = self.wo(self.b1a, 0, 4)
        b = self.wo(self.b1b, 2, 4)   # beririsan dengan a, bearing berbeda, mesin sama
        c = self.wo(self.b1a, 6, 4)   # menempel di ujung b (end == start) -> BUKAN konflik
        bars = self.bars(self.get())
        self.assertTrue(bars[a.id]["in_conflict"])
        self.assertEqual(bars[a.id]["conflicts_with"], [b.id])
        self.assertTrue(bars[b.id]["in_conflict"])
        self.assertEqual(bars[b.id]["conflicts_with"], [a.id])
        self.assertFalse(bars[c.id]["in_conflict"])
        self.assertEqual(bars[c.id]["conflicts_with"], [])

    def test_no_conflict_across_machines(self):
        a = self.wo(self.b1a, 0, 4)
        d = self.wo(self.b2, 0, 4)    # waktu sama, mesin lain
        bars = self.bars(self.get())
        self.assertFalse(bars[a.id]["in_conflict"])
        self.assertFalse(bars[d.id]["in_conflict"])

    def test_conflict_not_affected_by_priority_filter(self):
        a = self.wo(self.b1a, 0, 4, priority="HIGH")
        b = self.wo(self.b1b, 2, 4, priority="LOW")
        bars = self.bars(self.get(priority="HIGH"))
        self.assertEqual(set(bars), {a.id})
        self.assertTrue(bars[a.id]["in_conflict"])
        self.assertEqual(bars[a.id]["conflicts_with"], [b.id])

    def test_conflict_with_wo_outside_date_range(self):
        a = self.wo(self.b1a, 2, 6)
        b = self.wo(self.b1b, 0, 3)   # berakhir tepat di awal rentang -> di luar rentang
        rs = self.base + timedelta(hours=3)
        bars = self.bars(self.get(start=iso(rs), end=iso(rs + timedelta(days=1))))
        self.assertEqual(set(bars), {a.id})
        self.assertEqual(bars[a.id]["conflicts_with"], [b.id])

    # ---- filter
    def test_machine_filter_and_empty_rows(self):
        self.wo(self.b1a, 0, 2)
        self.wo(self.b2, 0, 2)
        r = self.get()
        names = {m["name"]: m for m in r.json()["machines"]}
        self.assertIn("Gantt M3 Kosong", names)               # mesin tanpa WO tetap jadi baris
        self.assertEqual(names["Gantt M3 Kosong"]["work_orders"], [])
        r = self.get(machine=str(self.m1.id))
        self.assertEqual([m["name"] for m in r.json()["machines"]], ["Gantt M1"])
        r = self.get(machine=f"{self.m1.id},{self.m2.id}")
        self.assertEqual({m["name"] for m in r.json()["machines"]}, {"Gantt M1", "Gantt M2"})
        self.client.force_authenticate(self.admin)
        r = self.client.get(URL, {"machine": [str(self.m1.id), str(self.m2.id)],
                                  "start": iso(self.base - timedelta(days=1)), "end": iso(self.base + timedelta(days=2))})
        self.assertEqual({m["name"] for m in r.json()["machines"]}, {"Gantt M1", "Gantt M2"})

    def test_priority_and_status_filters(self):
        hi = self.wo(self.b1a, 0, 2, priority="URGENT", status="IN_PROGRESS")
        lo = self.wo(self.b2, 0, 2, priority="LOW", status="SCHEDULED")
        self.assertEqual(set(self.bars(self.get(priority="URGENT"))), {hi.id})
        self.assertEqual(set(self.bars(self.get(priority="LOW,URGENT"))), {hi.id, lo.id})
        self.assertEqual(set(self.bars(self.get(status="IN_PROGRESS"))), {hi.id})
        self.assertEqual(set(self.bars(self.get(status="SCHEDULED"))), {lo.id})

    def test_invalid_params_400_rfc7807(self):
        for params in (
            {"status": "COMPLETED"},      # di luar {SCHEDULED, IN_PROGRESS}
            {"status": "scheduled"},      # enum harus persis
            {"priority": "CRITICAL"},
            {"machine": "bukan-uuid"},
            {"start": "kemarin"},
            {"start": iso(self.base), "end": iso(self.base - timedelta(days=1))},
        ):
            self.client.force_authenticate(self.admin)
            r = self.client.get(URL, params)
            self.assertEqual(r.status_code, 400, params)
            body = r.json()
            for k in ("type", "title", "status", "detail"):
                self.assertIn(k, body)
            self.assertEqual(body["status"], 400)

    # ---- isi bar
    def test_bar_label_technician_and_priority_rank(self):
        w = self.wo(self.b1a, 0, 2, priority="URGENT", tech=self.technician)
        w2 = self.wo(self.b2, 0, 2, priority="LOW")
        bars = self.bars(self.get())
        self.assertEqual(bars[w.id]["label"], "drive-end — g_tech")
        self.assertEqual(bars[w.id]["assigned_technician"]["name"], "g_tech")
        self.assertEqual(bars[w2.id]["label"], "drive-end")
        self.assertIsNone(bars[w2.id]["assigned_technician"])
        self.assertEqual(bars[w2.id]["priority_rank"], 0)
        self.assertEqual(bars[w.id]["priority_rank"], 3)
        self.assertEqual(bars[w.id]["bearing_position_label"], "drive-end")
