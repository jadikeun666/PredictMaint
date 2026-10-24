"""GET /bearings/{id}/derived-features/ dan /predictions-trend/ + pembaca InfluxDB bersama."""
import uuid
from datetime import datetime, timedelta, timezone as dt_timezone
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase
from django.utils import timezone
from rest_framework.test import APITestCase

from apps.registry.models import Bearing, Machine
from config import influx_queries
from config.api_errors import ServiceUnavailable

User = get_user_model()
DF_FN = "apps.registry.timeseries_views.query_derived_features"
PT_FN = "apps.registry.timeseries_views.query_predictions_trend"
T0 = datetime(2026, 9, 1, 0, 0, tzinfo=dt_timezone.utc)


def iso(dt):
    return dt.astimezone(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TimeseriesApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.users = [
            User.objects.create_user(username=f"ts_{r.lower()}", password="x", role=r)
            for r in ("ADMIN", "ENGINEER", "TECHNICIAN", "VIEWER")
        ]
        m = Machine.objects.create(name="Ts Rig", location="L", machine_type="t", criticality="LOW")
        cls.bearing = Bearing.objects.create(machine=m, position_label="de", bearing_model="X", status="ACTIVE")
        cls.df_url = f"/api/v1/bearings/{cls.bearing.id}/derived-features/"
        cls.pt_url = f"/api/v1/bearings/{cls.bearing.id}/predictions-trend/"

    # ---- akses & 404
    def test_unauthenticated_401(self):
        self.assertEqual(self.client.get(self.df_url).status_code, 401)
        self.assertEqual(self.client.get(self.pt_url).status_code, 401)

    def test_all_roles_200_and_shape(self):
        point = {"captured_at": T0, "rms": 0.5}
        with mock.patch(DF_FN, return_value=([point], False)), mock.patch(PT_FN, return_value=([], True)):
            for u in self.users:
                self.client.force_authenticate(u)
                r = self.client.get(self.df_url)
                self.assertEqual(r.status_code, 200, u.role)
                body = r.json()
                for k in ("bearing_id", "start", "end", "limit", "count", "truncated", "points"):
                    self.assertIn(k, body)
                self.assertEqual((body["count"], body["truncated"]), (1, False))
                r = self.client.get(self.pt_url)
                self.assertEqual(r.status_code, 200, u.role)
                self.assertEqual(r.json()["truncated"], True)
                self.assertIsNone(r.json()["prediction_type"])

    def test_unknown_bearing_404_and_non_uuid_path_404(self):
        self.client.force_authenticate(self.users[0])
        with mock.patch(DF_FN, return_value=([], False)):
            r = self.client.get(f"/api/v1/bearings/{uuid.uuid4()}/derived-features/")
            self.assertEqual((r.status_code, r.json()["code"]), (404, "not_found"))
            r = self.client.get("/api/v1/bearings/bukan-uuid/derived-features/")
            self.assertEqual(r.status_code, 404)

    # ---- rentang & limit
    def test_default_window_is_last_24h(self):
        self.client.force_authenticate(self.users[1])
        with mock.patch(DF_FN, return_value=([], False)) as fn:
            self.client.get(self.df_url)
        kw = fn.call_args.kwargs
        self.assertEqual(kw["end"] - kw["start"], timedelta(hours=24))
        self.assertLess(abs((timezone.now() - kw["end"]).total_seconds()), 30)
        self.assertEqual(kw["limit"], influx_queries.DEFAULT_LIMIT)
        self.assertEqual(kw["bearing_id"], self.bearing.id)

    def test_explicit_window_limit_and_type_forwarded(self):
        self.client.force_authenticate(self.users[1])
        params = {"start": iso(T0), "end": iso(T0 + timedelta(days=2)), "limit": "100"}
        with mock.patch(DF_FN, return_value=([], False)) as fn:
            self.client.get(self.df_url, params)
        kw = fn.call_args.kwargs
        self.assertEqual((kw["start"], kw["end"], kw["limit"]), (T0, T0 + timedelta(days=2), 100))
        with mock.patch(PT_FN, return_value=([], False)) as fn:
            r = self.client.get(self.pt_url, {**params, "prediction_type": "RUL"})
        self.assertEqual(fn.call_args.kwargs["prediction_type"], "RUL")
        self.assertEqual(r.json()["prediction_type"], "RUL")

    def test_invalid_params_400(self):
        self.client.force_authenticate(self.users[1])
        bad = [
            {"limit": "0"}, {"limit": "20001"}, {"limit": "abc"}, {"start": "kemarin"},
            {"start": iso(T0), "end": iso(T0 - timedelta(days=1))},
        ]
        with mock.patch(DF_FN, return_value=([], False)), mock.patch(PT_FN, return_value=([], False)):
            for p in bad:
                for url in (self.df_url, self.pt_url):
                    r = self.client.get(url, p)
                    self.assertEqual(r.status_code, 400, (url, p))
                    self.assertEqual(r.json()["code"], "invalid_query_param")
            for p in ({"prediction_type": "rul"}, {"prediction_type": "RUL,CLASSIFICATION"}):
                self.assertEqual(self.client.get(self.pt_url, p).status_code, 400, p)

    def test_influx_unavailable_maps_to_503(self):
        self.client.force_authenticate(self.users[1])
        with mock.patch(DF_FN, side_effect=ServiceUnavailable("x")):
            r = self.client.get(self.df_url)
        self.assertEqual((r.status_code, r.json()["code"]), (503, "service_unavailable"))


class _Rec:
    def __init__(self, t, **values):
        self._t, self.values = t, {"_time": t, **values}

    def get_time(self):
        return self._t


class _Client:
    def __init__(self, rows=None, fail=False):
        self.rows, self.fail, self.flux, self.closed = rows or [], fail, None, False

    def query_api(self):
        return self

    def query(self, flux):
        if self.fail:
            raise ConnectionError("down")
        self.flux = flux
        return [mock.Mock(records=self.rows)]

    def close(self):
        self.closed = True


class InfluxQueryUnitTests(SimpleTestCase):
    BID = uuid.uuid4()

    def run_q(self, fn, client, **kw):
        with mock.patch("config.influx_queries.get_influx_client", return_value=client):
            return fn(bearing_id=self.BID, start=T0, end=T0 + timedelta(days=1), **kw)

    def test_newest_kept_returned_ascending_and_truncated(self):
        t = [T0 + timedelta(minutes=i) for i in range(3)]
        client = _Client([_Rec(t[2], rms=3.0), _Rec(t[1], rms=2.0), _Rec(t[0], rms=1.0)])  # menurun
        points, truncated = self.run_q(influx_queries.query_derived_features, client, limit=2)
        self.assertTrue(truncated)
        self.assertEqual([p["rms"] for p in points], [2.0, 3.0])        # terbaru dipertahankan, menaik
        self.assertEqual(points[0]["captured_at"], t[1])
        self.assertIsNone(points[0]["bpfo_amp"])                        # field tak tertulis -> None
        self.assertIn(f'r.bearing_id == "{self.BID}"', client.flux)
        self.assertIn('r._measurement == "derived_features"', client.flux)
        self.assertIn("desc: true", client.flux)
        self.assertIn("limit(n: 3)", client.flux)                       # limit + 1
        self.assertIn("stop: 2026-09-02T00:00:00.000000Z", client.flux)
        self.assertTrue(client.closed)

    def test_not_truncated_when_within_limit(self):
        client = _Client([_Rec(T0, rms=1.0)])
        points, truncated = self.run_q(influx_queries.query_derived_features, client, limit=5)
        self.assertEqual((len(points), truncated), (1, False))

    def test_predictions_trend_shape_and_type_filter(self):
        client = _Client([_Rec(T0, prediction_type="RUL", model_version="m1", predicted_rul_hours=12.5)])
        points, _ = self.run_q(influx_queries.query_predictions_trend, client, limit=5, prediction_type="RUL")
        self.assertEqual(points[0]["predicted_rul_hours"], 12.5)
        self.assertIsNone(points[0]["fault_class"])
        self.assertEqual(points[0]["model_version"], "m1")
        self.assertIn('r.prediction_type == "RUL"', client.flux)
        self.assertIn('"model_version", "prediction_type"', client.flux)

    def test_invalid_prediction_type_rejected_before_query(self):
        with self.assertRaises(ValueError):
            self.run_q(influx_queries.query_predictions_trend, _Client(), limit=5, prediction_type='X" or true')

    def test_connection_failure_becomes_service_unavailable(self):
        client = _Client(fail=True)
        with self.assertLogs("config.influx_queries", level="ERROR"):
            with self.assertRaises(ServiceUnavailable):
                self.run_q(influx_queries.query_derived_features, client, limit=5)
        self.assertTrue(client.closed)
