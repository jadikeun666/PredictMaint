"""API /export-jobs/ (exports.md, api.md): akses, validasi param, 202, polling, unduh."""
import tempfile
from datetime import timedelta, timezone as dt_timezone
from datetime import datetime
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import override_settings
from rest_framework.test import APITestCase

from apps.registry.models import Bearing, Machine

from .models import ExportJob
from .storage import export_dir

User = get_user_model()
URL = "/api/v1/export-jobs/"
CELERY = "apps.exports.views.celery_app"
T0 = datetime(2026, 9, 1, tzinfo=dt_timezone.utc)


def iso(dt):
    return dt.astimezone(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ExportApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin, cls.engineer, cls.technician, cls.viewer = (
            User.objects.create_user(username=f"ex_{r.lower()}", password="x", role=r)
            for r in ("ADMIN", "ENGINEER", "TECHNICIAN", "VIEWER")
        )
        cls.m1 = Machine.objects.create(name="Ex Rig 1", location="L", machine_type="t", criticality="LOW")
        cls.m2 = Machine.objects.create(name="Ex Rig 2", location="L", machine_type="t", criticality="LOW")
        cls.b1 = Bearing.objects.create(machine=cls.m1, position_label="de", bearing_model="X", status="ACTIVE")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(override_settings(RAW_WAVEFORM_STORAGE_PATH=self.tmp.name))

    def window(self):
        return {"start": iso(T0), "end": iso(T0 + timedelta(days=7))}

    def hist(self, **extra):
        return {"export_type": "HISTORY_EXCEL", "params": {**self.window(), **extra}}

    def diag(self, **extra):
        return {"export_type": "DIAGNOSTIC_PDF", "params": {**self.window(), "bearing_id": str(self.b1.id), **extra}}

    def make_job(self, status="PENDING", export_type="HISTORY_EXCEL", file_path=None, user=None):
        return ExportJob.objects.create(
            export_type=export_type, requested_by=user or self.engineer, params=self.window(),
            status=status, file_path=file_path,
        )

    # ---- akses
    def test_unauthenticated_401(self):
        j = self.make_job()
        for call in (
            lambda: self.client.get(URL),
            lambda: self.client.post(URL, self.hist(), format="json"),
            lambda: self.client.get(f"{URL}{j.id}/"),
            lambda: self.client.get(f"{URL}{j.id}/download/"),
        ):
            self.assertEqual(call().status_code, 401)

    def test_create_allowed_only_for_engineer_and_admin(self):
        with mock.patch(CELERY):
            for u in (self.engineer, self.admin):
                self.client.force_authenticate(u)
                for payload in (self.hist(), self.diag()):
                    self.assertEqual(self.client.post(URL, payload, format="json").status_code, 202, (u.role, payload["export_type"]))
            before = ExportJob.objects.count()
            for u in (self.technician, self.viewer):
                self.client.force_authenticate(u)
                for payload in (self.hist(), self.diag()):
                    self.assertEqual(self.client.post(URL, payload, format="json").status_code, 403, (u.role, payload["export_type"]))
            self.assertEqual(ExportJob.objects.count(), before)

    def test_all_roles_can_list_and_retrieve(self):
        j = self.make_job()
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            self.assertEqual(self.client.get(URL).status_code, 200, u.role)
            self.assertEqual(self.client.get(f"{URL}{j.id}/").status_code, 200, u.role)

    def test_no_update_or_delete_405(self):
        j = self.make_job()
        self.client.force_authenticate(self.admin)
        for method in ("put", "patch", "delete", "post"):
            self.assertEqual(getattr(self.client, method)(f"{URL}{j.id}/").status_code, 405, method)
        self.assertTrue(ExportJob.objects.filter(pk=j.pk).exists())

    # ---- POST -> 202
    def test_create_returns_202_enqueues_on_export_queue(self):
        self.client.force_authenticate(self.engineer)
        with mock.patch(CELERY) as celery:
            r = self.client.post(URL, self.hist(), format="json")
        self.assertEqual(r.status_code, 202, r.content)
        body = r.json()
        self.assertEqual(
            set(body),
            {"id", "export_type", "requested_by", "params", "status", "requested_at", "completed_at", "download_url"},
        )
        self.assertNotIn("file_path", body)
        self.assertEqual((body["status"], body["completed_at"], body["download_url"]), ("PENDING", None, None))
        self.assertEqual(body["requested_by"], str(self.engineer.id))
        self.assertEqual(r["Location"], f"{URL}{body['id']}/")
        celery.send_task.assert_called_once_with("apps.exports.tasks.run_export", args=[body["id"]], queue="export")
        job = ExportJob.objects.get(pk=body["id"])
        self.assertEqual((job.status, job.requested_by_id), ("PENDING", self.engineer.id))

    def test_params_are_normalized_and_deduplicated(self):
        self.client.force_authenticate(self.engineer)
        payload = {"export_type": "HISTORY_EXCEL", "params": {
            "start": "2026-09-01T07:00:00+07:00", "end": "2026-09-08T00:00:00Z",
            "machine_ids": [str(self.m2.id), str(self.m1.id), str(self.m1.id)],
        }}
        with mock.patch(CELERY):
            r = self.client.post(URL, payload, format="json")
        self.assertEqual(r.status_code, 202, r.content)
        params = ExportJob.objects.get(pk=r.json()["id"]).params
        self.assertEqual(params["start"], "2026-09-01T00:00:00Z")
        self.assertEqual(params["machine_ids"], sorted({str(self.m1.id), str(self.m2.id)}))
        self.assertNotIn("bearing_ids", params)

    def test_invalid_payloads_400_and_no_job_created(self):
        self.client.force_authenticate(self.engineer)
        bad = [
            ({"export_type": "CSV", "params": self.window()}, "export_type"),
            ({"export_type": "HISTORY_EXCEL"}, "params"),
            ({"export_type": "DIAGNOSTIC_PDF", "params": self.window()}, "params"),                     # bearing_id wajib
            (self.diag(bearing_id="00000000-0000-0000-0000-000000000000"), "params"),                    # bearing tak ada
            (self.hist(bearing_id=str(self.b1.id)), "params"),                                          # param tak dikenal
            ({"export_type": "HISTORY_EXCEL", "params": {"start": iso(T0), "end": iso(T0)}}, "params"),  # start >= end
            ({"export_type": "HISTORY_EXCEL", "params": {"start": "kemarin", "end": iso(T0)}}, "params"),
            (self.hist(machine_ids=["00000000-0000-0000-0000-000000000000"]), "params"),                 # mesin tak ada
            (self.hist(machine_ids=str(self.m1.id)), "params"),                                         # bukan list
        ]
        with mock.patch(CELERY) as celery:
            for payload, key in bad:
                r = self.client.post(URL, payload, format="json")
                self.assertEqual(r.status_code, 400, payload)
                body = r.json()
                self.assertEqual(body["code"], "validation_error")
                self.assertIn(key, body["errors"], payload)
            celery.send_task.assert_not_called()
        self.assertEqual(ExportJob.objects.count(), 0)

    def test_unknown_param_is_named_in_error(self):
        self.client.force_authenticate(self.engineer)
        r = self.client.post(URL, self.hist(foo=1), format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("foo", r.json()["errors"]["params"])

    def test_enqueue_failure_marks_job_failed_and_returns_503(self):
        self.client.force_authenticate(self.engineer)
        with mock.patch(CELERY) as celery:
            celery.send_task.side_effect = RuntimeError("broker down")
            with self.assertLogs("apps.exports.views", level="ERROR"):
                r = self.client.post(URL, self.hist(), format="json")
        self.assertEqual((r.status_code, r.json()["code"]), (503, "service_unavailable"))
        self.assertEqual(ExportJob.objects.get().status, "FAILED")

    # ---- list
    def test_list_filters_and_invalid_filters(self):
        a = self.make_job("PENDING", "HISTORY_EXCEL")
        b = self.make_job("COMPLETED", "DIAGNOSTIC_PDF")
        c = self.make_job("FAILED", "HISTORY_EXCEL")
        self.client.force_authenticate(self.viewer)

        def ids(**p):
            return {x["id"] for x in self.client.get(URL, p).json()["results"]}

        self.assertEqual(ids(), {str(a.id), str(b.id), str(c.id)})
        self.assertEqual(ids(status="COMPLETED"), {str(b.id)})
        self.assertEqual(ids(export_type="HISTORY_EXCEL"), {str(a.id), str(c.id)})
        self.assertEqual(ids(status="FAILED,PENDING", export_type="HISTORY_EXCEL"), {str(a.id), str(c.id)})
        for p in ({"status": "DONE"}, {"export_type": "CSV"}):
            r = self.client.get(URL, p)
            self.assertEqual((r.status_code, r.json()["code"]), (400, "invalid_query_param"))

    def test_download_url_only_when_completed(self):
        pending = self.make_job("PENDING")
        done = self.make_job("COMPLETED", file_path=str(export_dir() / "x.xlsx"))
        self.client.force_authenticate(self.viewer)
        self.assertIsNone(self.client.get(f"{URL}{pending.id}/").json()["download_url"])
        self.assertEqual(self.client.get(f"{URL}{done.id}/").json()["download_url"], f"{URL}{done.id}/download/")

    # ---- unduh
    def test_download_completed_for_every_role(self):
        path = export_dir() / "hasil.xlsx"
        path.write_bytes(b"PK-isi-berkas")
        job = self.make_job("COMPLETED", "HISTORY_EXCEL", file_path=str(path))
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            r = self.client.get(f"{URL}{job.id}/download/")
            self.assertEqual(r.status_code, 200, u.role)
            self.assertEqual(b"".join(r.streaming_content), b"PK-isi-berkas")
            self.assertIn("attachment", r["Content-Disposition"])
            self.assertIn(f"history_excel-{job.id}.xlsx", r["Content-Disposition"])
            self.assertTrue(r["Content-Type"].startswith("application/vnd.openxmlformats"))

    def test_download_pdf_content_type(self):
        path = export_dir() / "hasil.pdf"
        path.write_bytes(b"%PDF-1.7 x")
        job = self.make_job("COMPLETED", "DIAGNOSTIC_PDF", file_path=str(path))
        self.client.force_authenticate(self.viewer)
        r = self.client.get(f"{URL}{job.id}/download/")
        self.assertEqual((r.status_code, r["Content-Type"]), (200, "application/pdf"))

    def test_download_not_ready_409(self):
        self.client.force_authenticate(self.viewer)
        for st in ("PENDING", "RUNNING", "FAILED"):
            job = self.make_job(st)
            r = self.client.get(f"{URL}{job.id}/download/")
            self.assertEqual((r.status_code, r.json()["code"]), (409, "conflict"), st)

    def test_download_missing_or_foreign_file_404(self):
        self.client.force_authenticate(self.viewer)
        gone = self.make_job("COMPLETED", file_path=str(export_dir() / "hilang.xlsx"))
        self.assertEqual(self.client.get(f"{URL}{gone.id}/download/").status_code, 404)
        with tempfile.NamedTemporaryFile(delete=False) as outside:   # file nyata di LUAR folder export
            outside.write(b"rahasia")
        self.addCleanup(lambda: __import__("os").unlink(outside.name))
        foreign = self.make_job("COMPLETED", file_path=outside.name)
        r = self.client.get(f"{URL}{foreign.id}/download/")
        self.assertEqual((r.status_code, r.json()["code"]), (404, "not_found"))
        nopath = self.make_job("COMPLETED", file_path=None)
        self.assertEqual(self.client.get(f"{URL}{nopath.id}/download/").status_code, 404)
