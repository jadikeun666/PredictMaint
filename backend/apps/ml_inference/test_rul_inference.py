"""
apps/ml_inference/test_rul_inference.py — RUL inference online (Sesi C, Giliran 2, g2b).
Bagian mock berjalan di mana saja; bagian artefak asli di-skip bila torch tidak ada.
"""
import os
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

import numpy as np
from django.test import SimpleTestCase

from apps.ml_inference import rul_inference as ri
from apps.ml_inference.feature_extraction import FEATURE_ORDER

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ORDER = list(FEATURE_ORDER)


class _Retry(Exception):
    pass


def _records(n, rms=0.1, drop=None):
    out = []
    for i in range(n):
        r = {"time": i, **{k: 1.0 + i * 0.01 for k in ORDER}}
        r["rms"] = rms
        if drop is not None and i == n - 1:
            r[drop] = None
        out.append(r)
    return out


class PureTests(SimpleTestCase):
    def test_insufficient_history(self):
        X, why = ri.build_window_matrix(_records(5), ORDER, 6)
        self.assertIsNone(X)
        self.assertEqual(why, "insufficient_history")

    def test_incomplete_features(self):
        X, why = ri.build_window_matrix(_records(8, drop="bpfo_amp"), ORDER, 6)
        self.assertIsNone(X)
        self.assertEqual(why, "incomplete_features")

    def test_uses_last_n_in_order(self):
        X, why = ri.build_window_matrix(_records(9), ORDER, 6)
        self.assertIsNone(why)
        self.assertEqual(X.shape, (6, len(ORDER)))
        self.assertAlmostEqual(X[0, ORDER.index("kurtosis")], 1.0 + 3 * 0.01)
        self.assertAlmostEqual(X[-1, ORDER.index("kurtosis")], 1.0 + 8 * 0.01)

    def test_dead_window_detection(self):
        X, _ = ri.build_window_matrix(_records(6, rms=0.1), ORDER, 6)
        self.assertFalse(ri.is_dead_window(X, ORDER))
        X2, _ = ri.build_window_matrix(_records(6, rms=0.001), ORDER, 6)
        self.assertTrue(ri.is_dead_window(X2, ORDER))

    def test_scale_window_shape_and_formula(self):
        X, _ = ri.build_window_matrix(_records(6), ORDER, 6)
        mean, std = np.arange(len(ORDER), dtype=float) * 0.1, np.arange(len(ORDER), dtype=float) + 1
        Z = ri.scale_window(X, mean, std)
        self.assertEqual(Z.shape, (1, 6, len(ORDER)))
        np.testing.assert_allclose(Z[0], (np.sign(X) * np.log1p(np.abs(X)) - mean) / std)

    def test_alerting_is_disabled_by_decision(self):
        self.assertIs(ri.RUL_ALERTING_ENABLED, False)

    def test_parse_time(self):
        a = ri.parse_time("2026-10-28T01:00:00.123456Z")
        b = ri.parse_time("2026-10-28T01:00:00.123456+00:00")
        self.assertEqual(a, b)
        self.assertIsNotNone(a.tzinfo)

    def test_history_bounds_has_no_look_ahead(self):
        at = ri.parse_time("2026-10-28T01:00:00.500000Z")
        start, stop = ri.history_bounds(at)
        self.assertEqual(stop - at, timedelta(microseconds=1))    # titik saat ini ikut, tidak ada yang lebih baru
        self.assertEqual(at - start, timedelta(days=ri.HISTORY_LOOKBACK_DAYS))
        self.assertLess(stop - at, timedelta(milliseconds=1))     # regresi: dulu +1 detik menarik titik masa depan


def _loaded(band=300.0):
    entry = {"model_version": "lstm-rul-test", "rul_confidence_band_hours": band}
    info = {"feature_order": ORDER, "seq_len": 6, "meta": {"model_version": "lstm-rul-test"}}
    return entry, object(), {"mu": 0.0, "sd": 1.0}, np.zeros(len(ORDER)), np.ones(len(ORDER)), info


class RunOrchestrationTests(SimpleTestCase):
    def _run(self, records=None, loaded="ok", sensor="s1", dup=False, pred=123.456,
             total=10, between=6, task=None):
        def count(sensor_id, start, stop):
            return total if start is None else between
        patches = {
            "_get_model": mock.patch.object(ri, "_get_model", return_value=(_loaded() if loaded == "ok" else None)),
            "_sensor_id_for_window": mock.patch.object(ri, "_sensor_id_for_window", return_value=sensor),
            "_already_predicted": mock.patch.object(ri, "_already_predicted", return_value=dup),
            "_count_windows": mock.patch.object(ri, "_count_windows", side_effect=count),
            "fetch_recent_features": mock.patch.object(ri, "fetch_recent_features",
                                                       return_value=records if records is not None else _records(6)),
            "_predict": mock.patch.object(ri, "_predict", return_value=np.array([pred])),
            "_persist": mock.patch.object(ri, "_persist", return_value="pred-uuid"),
            "_current_task": mock.patch.object(ri, "_current_task", return_value=task),
        }
        started = {k: p.start() for k, p in patches.items()}
        self.addCleanup(mock.patch.stopall)
        out = ri.run("b1", "w1", "2026-10-28T01:00:00Z")
        return out, started

    def test_happy_path_persists_with_band(self):
        out, m = self._run()
        self.assertEqual(out["status"], "predicted")
        self.assertAlmostEqual(out["predicted_rul_hours"], 123.456)
        self.assertAlmostEqual(out["band_low_hours"], 0.0)
        self.assertAlmostEqual(out["band_high_hours"], 423.456)
        m["_persist"].assert_called_once()
        args = m["_persist"].call_args[0]
        self.assertEqual((args[0], args[1], args[6]), ("b1", "w1", "lstm-rul-test"))

    def test_truly_insufficient_history_skips_without_influx_query(self):
        out, m = self._run(total=4)
        self.assertEqual((out["status"], out["reason"]), ("skipped", "insufficient_history"))
        m["fetch_recent_features"].assert_not_called()
        m["_persist"].assert_not_called()

    def test_in_flight_windows_are_not_treated_as_insufficient(self):
        # 10 window sudah masuk tetapi baru 4 titik fitur tertulis -> belum mapan (bukan insufficient)
        out, m = self._run(records=_records(4), total=10)
        self.assertEqual(out["reason"], "history_not_settled")
        m["_persist"].assert_not_called()

    def test_window_count_mismatch_is_unsettled(self):
        # 6 titik fitur terbaca, tetapi ada 7 window pada rentang itu (satu masih diproses)
        out, m = self._run(between=7)
        self.assertEqual(out["reason"], "history_not_settled")
        m["_persist"].assert_not_called()

    def test_unsettled_inside_celery_task_retries(self):
        task = mock.Mock()
        task.request.retries = 0
        task.retry.side_effect = _Retry("retry")
        with self.assertRaises(_Retry):
            self._run(between=7, task=task)
        task.retry.assert_called_once_with(countdown=ri.SETTLE_RETRY_COUNTDOWN_S, max_retries=ri.SETTLE_MAX_RETRIES)

    def test_retries_exhausted_skips_instead_of_raising(self):
        task = mock.Mock()
        task.request.retries = ri.SETTLE_MAX_RETRIES
        out, m = self._run(between=7, task=task)
        self.assertEqual(out["reason"], "history_not_settled")
        task.retry.assert_not_called()
        m["_persist"].assert_not_called()

    def test_dead_signal_no_prediction(self):
        out, m = self._run(records=_records(6, rms=0.001))
        self.assertEqual(out["reason"], "dead_signal")
        m["_persist"].assert_not_called()
        m["_predict"].assert_not_called()

    def test_incomplete_features_no_prediction(self):
        out, m = self._run(records=_records(6, drop="bpfo_amp"))
        self.assertEqual(out["reason"], "incomplete_features")
        m["_persist"].assert_not_called()

    def test_duplicate_skips_before_any_query(self):
        out, m = self._run(dup=True)
        self.assertEqual(out["reason"], "duplicate")
        m["fetch_recent_features"].assert_not_called()
        m["_count_windows"].assert_not_called()
        m["_persist"].assert_not_called()

    def test_no_model_registered(self):
        out, m = self._run(loaded=None)
        self.assertEqual(out["reason"], "no_rul_model_registered")
        m["_persist"].assert_not_called()

    def test_window_not_found(self):
        out, m = self._run(sensor=None)
        self.assertEqual(out["reason"], "window_not_found")
        m["_persist"].assert_not_called()

    def test_prediction_is_capped(self):
        out, _ = self._run(pred=5e9)
        self.assertLessEqual(out["predicted_rul_hours"], ri.MAX_RUL_HOURS)
        self.assertLessEqual(out["band_high_hours"], ri.MAX_RUL_HOURS)


@unittest.skipUnless(HAS_TORCH, "torch tidak tersedia")
class RealArtifactTests(SimpleTestCase):
    def test_end_to_end_with_real_artifact(self):
        from apps.ml_inference import lstm_rul as lr
        rng = np.random.default_rng(0)
        Xtr = rng.normal(size=(60, 6, len(ORDER))).astype(np.float32)
        ytr = np.exp(3.0 + 0.5 * Xtr[:, -1, 0])
        model, ts, _ = lr.fit_lstm(Xtr, ytr, seed=1, hp={"hidden": 4, "epochs": 2})
        mean, std = np.zeros(len(ORDER)), np.ones(len(ORDER))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            lr.save_artifact(path, model, ts, mean, std, ORDER, 6, {"hidden": 4},
                             meta={"model_version": "lstm-rul-real-test"})
            entry = {"model_version": "lstm-rul-real-test", "artifact_path": path, "rul_confidence_band_hours": 50.0}
            ri._cache.clear()
            self.addCleanup(ri._cache.clear)
            recs = _records(6)
            with mock.patch.object(ri.model_registry, "get_latest_entry", return_value=entry), \
                 mock.patch.object(ri, "_sensor_id_for_window", return_value="s1"), \
                 mock.patch.object(ri, "_already_predicted", return_value=False), \
                 mock.patch.object(ri, "_count_windows", side_effect=lambda s, a, b: 10 if a is None else 6), \
                 mock.patch.object(ri, "_current_task", return_value=None), \
                 mock.patch.object(ri, "fetch_recent_features", return_value=recs), \
                 mock.patch.object(ri, "_persist", return_value="pred-uuid") as persist:
                out = ri.run("b1", "w1", "2026-10-28T01:00:00Z")
        X, _ = ri.build_window_matrix(recs, ORDER, 6)
        expected = float(lr.predict_lstm(model, ts, ri.scale_window(X, mean, std))[0])
        self.assertEqual(out["status"], "predicted")
        self.assertAlmostEqual(out["predicted_rul_hours"], expected, places=6)
        self.assertTrue(np.isfinite(out["predicted_rul_hours"]) and out["predicted_rul_hours"] >= 0)
        self.assertAlmostEqual(out["band_high_hours"] - out["predicted_rul_hours"], 50.0, places=6)
        persist.assert_called_once()

    def test_artifact_version_mismatch_is_rejected(self):
        from apps.ml_inference import lstm_rul as lr
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 6, len(ORDER))).astype(np.float32)
        model, ts, _ = lr.fit_lstm(X, np.exp(rng.normal(3, .2, 20)), seed=1, hp={"hidden": 4, "epochs": 1})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            lr.save_artifact(path, model, ts, np.zeros(len(ORDER)), np.ones(len(ORDER)), ORDER, 6, {"hidden": 4},
                             meta={"model_version": "lain"})
            ri._cache.clear()
            self.addCleanup(ri._cache.clear)
            entry = {"model_version": "lstm-rul-x", "artifact_path": path, "rul_confidence_band_hours": 1.0}
            with mock.patch.object(ri.model_registry, "get_latest_entry", return_value=entry):
                with self.assertRaises(RuntimeError):
                    ri._get_model()
