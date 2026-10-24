"""Generator DIAGNOSTIC_PDF (exports.md §2): konteks, escaping, grafik, degradasi, PDF nyata, task."""
import base64
import tempfile
from datetime import datetime, timedelta, timezone as dt_timezone
from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.alerting.models import Alert
from apps.ml_inference.models import Prediction
from apps.registry.models import Bearing, Machine
from apps.scheduling.models import WorkOrder
from config.api_errors import ServiceUnavailable

from . import pdf_report
from .models import ExportJob
from .pdf_report import build_context, build_diagnostic_pdf, render_html, render_rul_chart
from .storage import output_path
from .tasks import run_export

User = get_user_model()
T0 = datetime(2026, 9, 1, tzinfo=dt_timezone.utc)
T_END = T0 + timedelta(days=10)
NOW = T0 + timedelta(days=30)
QF = "apps.exports.pdf_report.query_predictions_trend"
EVIL = "<script>alert(1)</script>"


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def rul_points(n=5, version="lstm-test-v1"):
    return [
        {"predicted_at": T0 + timedelta(days=i), "prediction_type": "RUL", "model_version": version,
         "fault_class": None, "confidence": None, "predicted_rul_hours": 500.0 - 80 * i}
        for i in range(n)
    ], False


class PdfBase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="pdf_eng", password="x", role="ENGINEER")
        cls.resolver = User.objects.create_user(username="pdf_resolver", password="x", role="ENGINEER")
        cls.machine = Machine.objects.create(name=f"Rig {EVIL}", location="L", machine_type="t", criticality="LOW")
        cls.bearing = Bearing.objects.create(
            machine=cls.machine, position_label="drive-end", bearing_model=f"SKF {EVIL}", status="ACTIVE")
        cls.other = Bearing.objects.create(machine=cls.machine, position_label="fan-end", bearing_model="X", status="ACTIVE")

        def pred(fault, when, conf="0.900", version="rf-test-v1", bearing=None):
            return Prediction.objects.create(
                bearing=bearing or cls.bearing, prediction_type="CLASSIFICATION", fault_class=fault,
                classification_confidence=Decimal(conf), model_version=version, predicted_at=when)

        cls.healthy = pred("HEALTHY", T0 + timedelta(days=1))
        cls.f1 = pred("OUTER_RACE", T0 + timedelta(days=2), "0.950")
        cls.f2 = pred("OUTER_RACE", T0 + timedelta(days=3), "0.870")
        cls.unknown = pred("UNKNOWN", T0 + timedelta(days=4), "0.400", version="rf-test-v2")
        cls.before = pred("BALL", T0 - timedelta(days=2))
        cls.after = pred("INNER_RACE", T0 + timedelta(days=12), "0.990", version="rf-test-v3")   # di luar rentang, terbaru
        pred("BALL", T0 + timedelta(days=2), bearing=cls.other)                                  # bearing lain
        cls.rul = Prediction.objects.create(
            bearing=cls.bearing, prediction_type="RUL", predicted_rul_hours=Decimal("143.50"),
            rul_confidence_band_low_hours=Decimal("118.00"), rul_confidence_band_high_hours=Decimal("169.00"),
            model_version="lstm-test-v1", predicted_at=T0 + timedelta(days=6))

        cls.a_open = Alert.objects.create(bearing=cls.bearing, alert_type="FAULT_DETECTED", severity="CRITICAL",
                                          triggered_at=T0 + timedelta(days=2), last_seen_at=T0 + timedelta(days=3))
        cls.a_user = Alert.objects.create(bearing=cls.bearing, alert_type="RUL_THRESHOLD", severity="HIGH",
                                          triggered_at=T0 + timedelta(days=3), last_seen_at=T0 + timedelta(days=3),
                                          resolved_at=T0 + timedelta(days=4), resolved_by=cls.resolver)
        cls.a_sys = Alert.objects.create(bearing=cls.bearing, alert_type="SENSOR_OFFLINE", severity="HIGH",
                                         triggered_at=T0 + timedelta(days=5), last_seen_at=T0 + timedelta(days=5),
                                         resolved_at=T0 + timedelta(days=6))
        Alert.objects.create(bearing=cls.other, alert_type="FAULT_DETECTED", severity="HIGH",
                             triggered_at=T0 + timedelta(days=2), last_seen_at=T0 + timedelta(days=2))
        cls.wo = WorkOrder.objects.create(bearing=cls.bearing, status="COMPLETED", priority="URGENT", notes=EVIL)
        WorkOrder.objects.filter(pk=cls.wo.pk).update(created_at=T0 + timedelta(days=3))
        wo_out = WorkOrder.objects.create(bearing=cls.bearing, status="DRAFT", priority="LOW")
        WorkOrder.objects.filter(pk=wo_out.pk).update(created_at=T0 - timedelta(days=40))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(override_settings(RAW_WAVEFORM_STORAGE_PATH=self.tmp.name))

    def job(self, bearing=None):
        return ExportJob.objects.create(
            export_type="DIAGNOSTIC_PDF", requested_by=self.user,
            params={"bearing_id": str((bearing or self.bearing).id), "start": iso(T0), "end": iso(T_END)})

    def ctx(self, points=None, **kw):
        with mock.patch(QF, return_value=points if points is not None else rul_points()):
            return build_context(self.job(), now=NOW, **kw)


class ContextTests(PdfBase):
    def test_summary_is_latest_as_of_generation_not_limited_to_range(self):
        c = self.ctx()
        self.assertEqual(c["latest_cls"]["fault_class"], "INNER_RACE")        # self.after: di luar rentang tapi terbaru
        self.assertEqual(c["latest_cls"]["confidence"], "0.990")
        self.assertEqual(c["latest_rul"]["hours"], "143.50")
        self.assertEqual((c["latest_rul"]["low"], c["latest_rul"]["high"]), ("118.00", "169.00"))

    def test_summary_ignores_predictions_after_generation_time(self):
        with mock.patch(QF, return_value=rul_points()):
            c = build_context(self.job(), now=T0 + timedelta(days=3, hours=12))
        self.assertEqual(c["latest_cls"]["fault_class"], "OUTER_RACE")        # self.f2 (hari ke-3), bukan UNKNOWN hari ke-4
        self.assertIsNone(c["latest_rul"])                                   # RUL hari ke-6 belum ada saat itu

    def test_fault_history_excludes_healthy_includes_unknown_newest_first(self):
        c = self.ctx()
        self.assertEqual(c["class_total"], 4)                                 # healthy, f1, f2, unknown
        self.assertEqual(c["fault_total"], 3)
        self.assertEqual([r["fault_class"] for r in c["fault_rows"]], ["UNKNOWN", "OUTER_RACE", "OUTER_RACE"])
        self.assertEqual(c["fault_rows"][0]["confidence"], "0.400")
        self.assertFalse(c["fault_truncated"])

    def test_row_cap_keeps_newest_and_reports_total(self):
        with mock.patch.object(pdf_report, "MAX_TABLE_ROWS", 2):
            c = self.ctx()
        self.assertEqual(len(c["fault_rows"]), 2)
        self.assertEqual(c["fault_total"], 3)
        self.assertTrue(c["fault_truncated"])
        self.assertEqual(c["fault_rows"][0]["fault_class"], "UNKNOWN")

    def test_alerts_and_work_orders_scoped_to_bearing_and_range(self):
        c = self.ctx()
        self.assertEqual(c["alert_total"], 3)
        res = {a["alert_type"]: a["resolution"] for a in c["alert_rows"]}
        self.assertEqual(res["FAULT_DETECTED"], "Belum resolved")
        self.assertIn("oleh pdf_resolver", res["RUL_THRESHOLD"])
        self.assertIn("oleh sistem", res["SENSOR_OFFLINE"])
        self.assertEqual(c["wo_total"], 1)
        self.assertEqual(c["wo_rows"][0]["status"], "COMPLETED")

    def test_provenance_lists_every_model_version_behind_the_numbers(self):
        c = self.ctx(points=rul_points(version="lstm-test-v1"))
        self.assertEqual(c["model_versions"], ["lstm-test-v1", "rf-test-v1", "rf-test-v2", "rf-test-v3"])

    def test_influx_down_degrades_gracefully(self):
        with mock.patch(QF, side_effect=ServiceUnavailable("x")):
            with self.assertLogs("apps.exports.pdf_report", level="WARNING"):
                c = build_context(self.job(), now=NOW)
        self.assertIsNone(c["chart"])
        self.assertIn("InfluxDB", c["rul_error"])
        self.assertIn("InfluxDB", render_html(c))

    def test_no_rul_data_message(self):
        c = self.ctx(points=([], False))
        self.assertIsNone(c["chart"])
        self.assertIsNone(c["rul_error"])
        self.assertIn("Belum ada prediksi RUL pada rentang ini", render_html(c))

    def test_other_bearing_gets_its_own_data(self):
        with mock.patch(QF, return_value=([], False)):
            c = build_context(self.job(self.other), now=NOW)
        self.assertEqual(c["fault_total"], 1)
        self.assertEqual(c["alert_total"], 1)
        self.assertEqual(c["wo_total"], 0)


class RenderTests(PdfBase):
    def test_html_escapes_user_controlled_text(self):
        html = render_html(self.ctx())
        self.assertNotIn(EVIL, html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)

    def test_html_contains_required_sections_and_values(self):
        html = render_html(self.ctx())
        for needle in ("Laporan Diagnostik Bearing", "drive-end", "pdf_eng", "Riwayat fault", "Tren RUL",
                       "Alert dan work order", "Asal angka", "rf-test-v3", "143.50 jam", "data:image/png;base64,"):
            self.assertIn(needle, html, needle)

    def test_chart_is_png_data_uri(self):
        uri = render_rul_chart(rul_points(8)[0])
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1])[:8], b"\x89PNG\r\n\x1a\n")

    def test_chart_none_without_numeric_points(self):
        self.assertIsNone(render_rul_chart([]))
        self.assertIsNone(render_rul_chart([{"predicted_at": T0, "predicted_rul_hours": None}]))


class PdfFileTests(PdfBase):
    def test_real_pdf_is_written_atomically(self):
        job = self.job()
        with mock.patch(QF, return_value=rul_points(12)):
            path = build_diagnostic_pdf(job)
        self.assertEqual(path, output_path(job))
        data = path.read_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertIn(b"%%EOF", data[-1024:])
        self.assertGreater(len(data), 5000)
        self.assertFalse(list(path.parent.glob("*.part")))

    def test_pdf_still_builds_when_influx_is_down(self):
        job = self.job()
        with mock.patch(QF, side_effect=ServiceUnavailable("x")):
            path = build_diagnostic_pdf(job)
        self.assertTrue(path.read_bytes().startswith(b"%PDF-"))

    def test_task_completes_diagnostic_job(self):
        job = self.job()
        with mock.patch(QF, return_value=rul_points()):
            run_export(str(job.id))
        job.refresh_from_db()
        self.assertEqual(job.status, "COMPLETED")
        self.assertIsNotNone(job.completed_at)
        self.assertTrue(output_path(job).read_bytes().startswith(b"%PDF-"))

    def test_task_marks_failed_when_bearing_is_gone(self):
        job = self.job()
        ExportJob.objects.filter(pk=job.pk).update(params={**job.params, "bearing_id": "00000000-0000-0000-0000-000000000000"})
        with self.assertLogs("apps.exports.tasks", level="ERROR"):
            run_export(str(job.id))
        job.refresh_from_db()
        self.assertEqual((job.status, job.file_path, job.completed_at), ("FAILED", None, None))
