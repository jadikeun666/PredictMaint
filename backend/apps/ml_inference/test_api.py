"""API predictions: read-only, filter, paginasi (api.md)."""
from datetime import timedelta, timezone as dt_timezone
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APITestCase

from apps.registry.models import Bearing, Machine

from .models import Prediction

User = get_user_model()
URL = "/api/v1/predictions/"


def make_user(username, role):
    return User.objects.create_user(username=username, password="x", role=role)


def iso(dt):
    return dt.astimezone(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PredictionApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = make_user("pr_admin", User.Role.ADMIN)
        cls.engineer = make_user("pr_eng", User.Role.ENGINEER)
        cls.technician = make_user("pr_tech", User.Role.TECHNICIAN)
        cls.viewer = make_user("pr_view", User.Role.VIEWER)
        m = Machine.objects.create(name="Pr Rig", location="L", machine_type="t", criticality="LOW")
        cls.b1 = Bearing.objects.create(machine=m, position_label="drive-end", bearing_model="X", status="ACTIVE")
        cls.b2 = Bearing.objects.create(machine=m, position_label="fan-end", bearing_model="X", status="ACTIVE")
        cls.now = timezone.now().replace(microsecond=0)

    def pred(self, bearing=None, fault="OUTER_RACE", age_min=0, version="rf-test-v1", ptype="CLASSIFICATION"):
        kw = dict(
            bearing=bearing or self.b1, prediction_type=ptype, model_version=version,
            predicted_at=self.now - timedelta(minutes=age_min),
        )
        if ptype == "CLASSIFICATION":
            kw.update(fault_class=fault, classification_confidence=Decimal("0.950"))
        else:
            kw.update(predicted_rul_hours=Decimal("120.50"),
                      rul_confidence_band_low_hours=Decimal("100.00"),
                      rul_confidence_band_high_hours=Decimal("140.00"))
        return Prediction.objects.create(**kw)

    # ---- akses
    def test_unauthenticated_401(self):
        p = self.pred()
        self.assertEqual(self.client.get(URL).status_code, 401)
        self.assertEqual(self.client.get(f"{URL}{p.id}/").status_code, 401)

    def test_all_roles_read_with_exact_fields(self):
        p = self.pred()
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            r = self.client.get(URL)
            self.assertEqual(r.status_code, 200, u.role)
            self.assertIn("results", r.json())
            r = self.client.get(f"{URL}{p.id}/")
            self.assertEqual(r.status_code, 200, u.role)
            body = r.json()
            for f in ("id", "window", "bearing", "prediction_type", "fault_class",
                      "classification_confidence", "predicted_rul_hours",
                      "rul_confidence_band_low_hours", "rul_confidence_band_high_hours",
                      "model_version", "predicted_at"):
                self.assertIn(f, body)

    def test_no_write_endpoints_405(self):
        p = self.pred()
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.post(URL, {}, format="json").status_code, 405)
        for method in ("put", "patch", "delete"):
            self.assertEqual(getattr(self.client, method)(f"{URL}{p.id}/").status_code, 405, method)
        self.assertTrue(Prediction.objects.filter(id=p.id).exists())

    def test_classification_and_rul_shapes(self):
        c = self.pred(ptype="CLASSIFICATION")
        r_ = self.pred(ptype="RUL")
        self.client.force_authenticate(self.viewer)
        c_body = self.client.get(f"{URL}{c.id}/").json()
        self.assertEqual(c_body["classification_confidence"], "0.950")
        self.assertIsNone(c_body["predicted_rul_hours"])
        r_body = self.client.get(f"{URL}{r_.id}/").json()
        self.assertEqual(r_body["predicted_rul_hours"], "120.50")
        self.assertEqual(r_body["rul_confidence_band_low_hours"], "100.00")
        self.assertEqual(r_body["rul_confidence_band_high_hours"], "140.00")
        self.assertIsNone(r_body["fault_class"])
        self.assertIsNone(r_body["classification_confidence"])

    # ---- list
    def test_ordering_newest_first(self):
        old = self.pred(age_min=60)
        new = self.pred(age_min=1)
        mid = self.pred(age_min=30)
        self.client.force_authenticate(self.viewer)
        ids = [x["id"] for x in self.client.get(URL).json()["results"]]
        self.assertEqual(ids, [str(new.id), str(mid.id), str(old.id)])

    def test_cursor_pagination(self):
        for i in range(55):
            self.pred(age_min=i)
        self.client.force_authenticate(self.viewer)
        p1 = self.client.get(URL).json()
        self.assertEqual(len(p1["results"]), 50)
        self.assertIsNotNone(p1["next"])
        p2 = self.client.get(p1["next"]).json()
        self.assertEqual(len(p2["results"]), 5)
        self.assertIsNone(p2["next"])
        self.assertFalse({x["id"] for x in p1["results"]} & {x["id"] for x in p2["results"]})

    def test_filters(self):
        a = self.pred(self.b1, "OUTER_RACE", version="v1")
        b = self.pred(self.b2, "HEALTHY", version="v2")
        c = self.pred(self.b1, ptype="RUL", version="v2")
        self.client.force_authenticate(self.viewer)

        def ids(**p):
            return {x["id"] for x in self.client.get(URL, p).json()["results"]}

        self.assertEqual(ids(bearing=str(self.b2.id)), {str(b.id)})
        self.assertEqual(ids(prediction_type="RUL"), {str(c.id)})
        self.assertEqual(ids(fault_class="HEALTHY"), {str(b.id)})
        self.assertEqual(ids(fault_class="HEALTHY,OUTER_RACE"), {str(a.id), str(b.id)})
        self.assertEqual(ids(model_version="v2"), {str(b.id), str(c.id)})
        self.assertEqual(ids(bearing=str(self.b1.id), model_version="v2"), {str(c.id)})

    def test_time_range_start_inclusive_end_exclusive(self):
        p3 = self.pred(age_min=180)
        p2 = self.pred(age_min=120)
        p1 = self.pred(age_min=60)
        self.client.force_authenticate(self.viewer)

        def ids(**p):
            return {x["id"] for x in self.client.get(URL, p).json()["results"]}

        self.assertEqual(ids(start=iso(self.now - timedelta(minutes=120))), {str(p2.id), str(p1.id)})
        self.assertEqual(ids(end=iso(self.now - timedelta(minutes=120))), {str(p3.id)})
        self.assertEqual(
            ids(start=iso(self.now - timedelta(minutes=150)), end=iso(self.now - timedelta(minutes=90))),
            {str(p2.id)},
        )

    def test_invalid_params_400_rfc7807(self):
        self.client.force_authenticate(self.viewer)
        t = self.now
        for params in (
            {"prediction_type": "NOPE"},
            {"fault_class": "healthy"},
            {"bearing": "x"},
            {"start": "kemarin"},
            {"start": iso(t), "end": iso(t - timedelta(days=1))},
        ):
            r = self.client.get(URL, params)
            self.assertEqual(r.status_code, 400, params)
            body = r.json()
            for k in ("type", "title", "status", "detail"):
                self.assertIn(k, body)
            self.assertEqual(body["status"], 400)
