"""Tes penjaga provenance (Sesi E) + tes pertama untuk jalur tasks.run_inference (semua efek samping di-mock)."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from apps.ml_inference import provenance_guard as pg
from config import classification_provenance as cfg

MV = "rf-baseline-cwru-v1.1"
CWRU = "SKF 6205-2RS JEM (drive end)"
IMS = "Rexnord ZA-2115"


def _check(raw="INNER_RACE", conf=0.95, mv=MV, lookup=None):
    with patch.object(pg, "lookup_bearing_model", lookup or (lambda b: CWRU)):
        return pg.check(bearing_id="b1", model_version=mv, raw_class=raw, confidence=conf, threshold=0.6)


class ProvenanceGuardUnitTests(unittest.TestCase):
    def test_flag_default_is_off(self):
        self.assertFalse(cfg.CLASSIFICATION_PROVENANCE_GUARD_ENABLED)

    def test_disabled_never_rejects_and_never_looks_up(self):
        def boom(b):
            raise AssertionError("lookup tidak boleh dipanggil")
        with patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", False):
            self.assertFalse(_check(lookup=boom, mv="apa-saja").rejected)

    def test_enabled_allowed_bearing_passes(self):
        with patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", True):
            d = _check(lookup=lambda b: CWRU)
        self.assertFalse(d.rejected)
        self.assertEqual(d.bearing_model, CWRU)

    def test_enabled_disallowed_bearing_rejected(self):
        with patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", True):
            d = _check(lookup=lambda b: IMS)
        self.assertTrue(d.rejected)
        self.assertEqual(d.reason, "bearing_model_not_allowed")

    def test_match_is_exact_and_case_sensitive(self):
        with patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", True):
            self.assertTrue(_check(lookup=lambda b: CWRU.lower()).rejected)

    def test_unknown_or_missing_bearing_fail_closed(self):
        with patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", True):
            d = _check(lookup=lambda b: None)
        self.assertTrue(d.rejected)
        self.assertEqual(d.reason, "bearing_model_unknown")

    def test_unlisted_model_version_fail_closed_without_lookup(self):
        def boom(b):
            raise AssertionError("lookup tidak perlu")
        with patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", True):
            d = _check(mv="model-baru-v9", lookup=boom)
        self.assertTrue(d.rejected)
        self.assertEqual(d.reason, "model_version_not_in_allowlist")

    def test_only_fault_predictions_above_threshold_are_candidates(self):
        def boom(b):
            raise AssertionError("tidak boleh lookup")
        with patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", True):
            self.assertFalse(_check(raw="HEALTHY", conf=0.99, lookup=boom).rejected)
            self.assertFalse(_check(raw="UNKNOWN", conf=0.99, lookup=boom).rejected)
            self.assertFalse(_check(raw="BALL", conf=0.59, lookup=boom).rejected)
            self.assertTrue(_check(raw="BALL", conf=0.60, lookup=lambda b: IMS).rejected)

    def test_lookup_errors_propagate(self):
        def err(b):
            raise RuntimeError("db down")
        with patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", True):
            with self.assertRaises(RuntimeError):
                _check(lookup=err)

    def test_allowlist_is_validated_model_only(self):
        self.assertEqual(cfg.ALLOWED_BEARING_MODELS, {MV: (CWRU,)})

    def test_fault_classes_match_alerting(self):
        try:
            from apps.alerting.severity import FAULT_CLASSES
        except ImportError:
            self.skipTest("apps.alerting tidak tersedia di lingkungan ini")
        self.assertEqual(set(FAULT_CLASSES), set(pg.FAULT_CLASSES))


class FakeClf:
    classes_ = np.array(["BALL", "HEALTHY", "INNER_RACE", "OUTER_RACE"])

    def __init__(self, probs):
        self._p = np.array(probs, dtype=float)

    def predict_proba(self, X):
        return np.array([self._p])


FEATS = {"rms": 0.1, "kurtosis": 1.0, "crest_factor": 4.0, "skewness": 0.0, "bpfo_amp": 10.0,
         "bpfi_amp": 20.0, "bsf_amp": 5.0, "ftf_amp": 2.0, "dominant_freq_hz": 100.0}


class RunInferenceIntegrationTests(unittest.TestCase):
    """Jalur run_inference dengan semua efek samping di-mock (sebelumnya tanpa tes)."""

    def setUp(self):
        try:
            from apps.ml_inference import feature_extraction as fe
            from apps.ml_inference import tasks
        except ImportError as exc:
            self.skipTest("tasks tidak dapat diimpor di lingkungan ini: %s" % exc)
        self.tasks = tasks
        self.entry = {"model_version": MV, "feature_order": list(fe.MODEL_FEATURE_ORDER)}
        self.saved = []

        def fake_save(**kw):
            self.saved.append(kw)
            return SimpleNamespace(id="pid", predicted_at="t", fault_class=kw["fault_class"],
                                   classification_confidence=kw["confidence"])
        self.alert = patch.object(tasks, "_enqueue_alerting").start()
        self.rul = patch.object(tasks, "_enqueue_rul_inference").start()
        self.ts = patch.object(tasks, "write_prediction_ts").start()
        patch.object(tasks, "save_classification_prediction", side_effect=fake_save).start()
        self.addCleanup(patch.stopall)

    def _run(self, probs, features=None, bearing_model=CWRU, guard=False):
        clf = FakeClf(probs)
        patch.object(self.tasks, "_get_classifier", return_value=(clf, self.entry)).start()
        patch.object(pg, "lookup_bearing_model", return_value=bearing_model).start()
        patch.object(cfg, "CLASSIFICATION_PROVENANCE_GUARD_ENABLED", guard).start()
        return self.tasks.run_inference(bearing_id="b1", window_id="w1", captured_at="2026-11-10T00:00:00Z",
                                        features=features or dict(FEATS))

    # probs urutan classes_: BALL, HEALTHY, INNER_RACE, OUTER_RACE
    def test_guard_off_keeps_fault(self):
        r = self._run([0.01, 0.02, 0.95, 0.02], bearing_model=IMS, guard=False)
        self.assertEqual(r["fault_class"], "INNER_RACE")
        self.assertFalse(r["provenance_rejected"])

    def test_guard_on_rejects_foreign_bearing_keeps_raw_confidence(self):
        r = self._run([0.01, 0.02, 0.95, 0.02], bearing_model=IMS, guard=True)
        self.assertEqual(r["fault_class"], "UNKNOWN")
        self.assertTrue(r["provenance_rejected"])
        self.assertEqual(r["raw_predicted_class"], "INNER_RACE")
        self.assertAlmostEqual(self.saved[0]["confidence"], 0.95, places=6)
        self.assertEqual(self.saved[0]["fault_class"], "UNKNOWN")
        self.assertEqual(self.saved[0]["model_version"], MV)
        self.alert.assert_called_once()
        self.rul.assert_called_once()

    def test_guard_on_allows_validated_bearing(self):
        r = self._run([0.01, 0.02, 0.95, 0.02], bearing_model=CWRU, guard=True)
        self.assertEqual(r["fault_class"], "INNER_RACE")
        self.assertFalse(r["provenance_rejected"])

    def test_guard_on_does_not_touch_healthy_or_low_confidence(self):
        r = self._run([0.05, 0.90, 0.03, 0.02], bearing_model=IMS, guard=True)
        self.assertEqual(r["fault_class"], "HEALTHY")
        r2 = self._run([0.30, 0.10, 0.50, 0.10], bearing_model=IMS, guard=True)
        self.assertEqual(r2["fault_class"], "UNKNOWN")
        self.assertFalse(r2["provenance_rejected"])

    def test_missing_fault_frequency_features_still_skipped(self):
        feats = dict(FEATS)
        feats["bpfo_amp"] = None
        r = self._run([0.01, 0.02, 0.95, 0.02], features=feats, guard=True)
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(self.saved, [])


if __name__ == "__main__":
    unittest.main()
