"""Unit test severity (scheduling.md §1) — fungsi murni, tanpa DB."""
from django.test import SimpleTestCase

from apps.alerting.severity import compute_severity


class ClassificationSeverityTests(SimpleTestCase):
    CASES = [
        ("OUTER_RACE", 1.0, "CRITICAL"),
        ("INNER_RACE", 0.85, "CRITICAL"),   # batas bawah CRITICAL inklusif
        ("BALL", 0.8499, "HIGH"),
        ("OUTER_RACE", 0.6, "HIGH"),        # batas bawah HIGH inklusif
        ("OUTER_RACE", 0.5999, None),
        ("OUTER_RACE", None, None),
        ("HEALTHY", 1.0, None),
        ("UNKNOWN", 0.9, None),
    ]

    def test_table(self):
        for fault_class, conf, expected in self.CASES:
            with self.subTest(fault_class=fault_class, conf=conf):
                r = compute_severity(fault_class=fault_class, confidence=conf)
                self.assertEqual(r.severity if r else None, expected)
                if r:
                    self.assertEqual(r.alert_type, "FAULT_DETECTED")


class RulSeverityTests(SimpleTestCase):
    CASES = [
        (0, "CRITICAL"), (168, "CRITICAL"), (168.01, "HIGH"),
        (336, "HIGH"), (336.01, "WARNING"), (720, "WARNING"),
        (720.01, None), (None, None),
    ]

    def test_table(self):
        for rul, expected in self.CASES:
            with self.subTest(rul=rul):
                r = compute_severity(rul_hours=rul)
                self.assertEqual(r.severity if r else None, expected)
                if r:
                    self.assertEqual(r.alert_type, "RUL_THRESHOLD")


class CombinedSeverityTests(SimpleTestCase):
    def test_higher_governs_rul(self):
        r = compute_severity(fault_class="BALL", confidence=0.7, rul_hours=100)
        self.assertEqual((r.severity, r.alert_type), ("CRITICAL", "RUL_THRESHOLD"))

    def test_higher_governs_classification(self):
        r = compute_severity(fault_class="OUTER_RACE", confidence=0.9, rul_hours=500)
        self.assertEqual((r.severity, r.alert_type), ("CRITICAL", "FAULT_DETECTED"))

    def test_tie_goes_to_classification(self):
        r = compute_severity(fault_class="OUTER_RACE", confidence=0.9, rul_hours=100)
        self.assertEqual(r.alert_type, "FAULT_DETECTED")
