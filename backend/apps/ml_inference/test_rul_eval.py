"""
apps/ml_inference/test_rul_eval.py — test untuk rul_eval (numpy murni) dan
lstm_rul (torch; di-skip otomatis bila torch tidak ada).
"""
import unittest

import numpy as np
from django.test import SimpleTestCase

from apps.ml_inference import rul_eval as ev

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ScalerTests(SimpleTestCase):
    def test_signed_log1p_odd_and_zero(self):
        x = np.array([-3.0, 0.0, 3.0])
        y = ev.signed_log1p(x)
        self.assertAlmostEqual(y[0], -y[2])
        self.assertEqual(y[1], 0.0)

    def test_scaler_zscore_and_constant_guard(self):
        rng = np.random.default_rng(0)
        a = np.column_stack([rng.normal(5, 2, 200), np.full(200, 7.0)])
        sc = ev.fit_scaler([a])
        z = ev.apply_scaler(sc, a)
        self.assertAlmostEqual(float(z[:, 0].mean()), 0.0, places=9)
        self.assertAlmostEqual(float(z[:, 0].std()), 1.0, places=9)
        np.testing.assert_allclose(z[:, 1], 0.0, atol=1e-12)             # fitur konstan -> 0, bukan NaN
        self.assertTrue(np.all(np.isfinite(z)))


class WindowTests(SimpleTestCase):
    def test_shapes_alignment_and_gap_flag(self):
        n = 20
        feats = np.arange(n * 2, dtype=float).reshape(n, 2)
        ts = np.arange(n) * 600.0
        ts[10:] += 3600.0                                    # celah 70 menit antara titik 9 dan 10
        rul = np.linspace(19, 0, n)
        ws = ev.make_windows(feats, ts, rul, seq_len=6)
        self.assertEqual(ws.X.shape, (n - 5, 6, 2))
        self.assertEqual(ws.last_idx[0], 5)
        np.testing.assert_array_equal(ws.X[0], feats[0:6])
        np.testing.assert_array_equal(ws.X[-1], feats[n - 6:n])
        self.assertEqual(ws.y[0], rul[5])
        crossing = set(ws.last_idx[ws.crosses_gap].tolist())
        self.assertEqual(crossing, {10, 11, 12, 13, 14})     # jendela yang memuat pasangan (9,10)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            ev.make_windows(np.zeros((4, 2)), np.arange(4.0), np.zeros(4))


class MetricTests(SimpleTestCase):
    def test_perfect_prediction(self):
        t = np.array([10.0, 5.0, 1.0])
        m = ev.regression_metrics(t, t, traj_len_hours=10.0)
        self.assertEqual(m["mae"], 0.0)
        self.assertEqual(m["phm"], 0.0)

    def test_late_penalised_more_than_early(self):
        true = np.array([50.0])
        late = ev.regression_metrics(true + 10, true, 100.0)["phm"]     # terlalu optimis
        early = ev.regression_metrics(true - 10, true, 100.0)["phm"]
        self.assertGreater(late, early)
        self.assertAlmostEqual(late, np.exp(10 / 10) - 1)
        self.assertAlmostEqual(early, np.exp(10 / 13) - 1)

    def test_mae_rmse_bias_and_last10(self):
        true = np.array([100.0, 8.0, 4.0])
        pred = np.array([90.0, 10.0, 2.0])
        m = ev.regression_metrics(pred, true, traj_len_hours=100.0)
        self.assertAlmostEqual(m["mae"], (10 + 2 + 2) / 3)
        self.assertAlmostEqual(m["rmse"], np.sqrt((100 + 4 + 4) / 3))
        self.assertAlmostEqual(m["bias"], (-10 + 2 - 2) / 3)
        self.assertAlmostEqual(m["mae_last10pct"], 2.0)      # hanya true <= 10 jam


class BaselineTests(SimpleTestCase):
    def test_constant_is_train_mean(self):
        p = ev.baseline_constant([np.array([0.0, 10.0]), np.array([20.0])], 4)
        np.testing.assert_allclose(p, 10.0)

    def test_hi_extrapolation_exact_on_linear_degradation(self):
        n = 60
        ts = np.arange(n) * 600.0
        frac = np.arange(n) / (n - 1)
        raw = np.column_stack([1.0 + 4.0 * frac, 3.0 + 6.0 * frac])   # rms, kurtosis naik linear
        rul = (ts[-1] - ts) / 3600.0
        p = ev.fit_hi_params([raw], [rul], 0, 1)
        ws = ev.make_windows(raw, ts, rul)
        pred, fb = ev.hi_extrapolation_predict(raw, ts, ws.last_idx, p, 0, 1)
        self.assertFalse(fb.any())
        np.testing.assert_allclose(pred, ws.y, atol=1e-6)    # HI linear -> RUL terpulihkan persis

    def test_flat_signal_uses_fallback(self):
        n = 30
        ts = np.arange(n) * 600.0
        raw = np.column_stack([np.ones(n), np.ones(n)])
        rul = (ts[-1] - ts) / 3600.0
        p = ev.fit_hi_params([raw], [rul], 0, 1)
        ws = ev.make_windows(raw, ts, rul)
        pred, fb = ev.hi_extrapolation_predict(raw, ts, ws.last_idx, p, 0, 1)
        self.assertTrue(fb.all())
        np.testing.assert_allclose(pred, p.fallback_hours)


class FoldTests(SimpleTestCase):
    TRAJS = [("A", 1), ("B", 1), ("C", 2), ("D", 3)]

    def test_loto(self):
        folds = ev.build_folds(self.TRAJS, "loto")
        self.assertEqual(len(folds), 4)
        for name, test_ids, train_ids in folds:
            self.assertEqual(len(test_ids), 1)
            self.assertFalse(set(test_ids) & set(train_ids))
            self.assertEqual(len(train_ids), 3)

    def test_leave_set_out_holds_whole_set(self):
        folds = {n: (t, tr) for n, t, tr in ev.build_folds(self.TRAJS, "leave_set_out")}
        self.assertEqual(set(folds), {"set1", "set2", "set3"})
        self.assertEqual(sorted(folds["set1"][0]), ["A", "B"])
        self.assertEqual(sorted(folds["set1"][1]), ["C", "D"])

    def test_excl_set3_never_uses_D(self):
        for _, test_ids, train_ids in ev.build_folds(self.TRAJS, "loto_excl_set3"):
            self.assertNotIn("D", test_ids + train_ids)

    def test_unknown_scheme(self):
        with self.assertRaises(ValueError):
            ev.build_folds(self.TRAJS, "nope")


@unittest.skipUnless(HAS_TORCH, "torch tidak tersedia")
class LSTMTests(SimpleTestCase):
    def test_learns_simple_signal_and_is_deterministic(self):
        from apps.ml_inference.lstm_rul import fit_predict_lstm
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 6, 3)).astype(np.float32)
        y = np.exp(2.0 + 0.8 * X[:, -1, 0])                    # RUL (jam) fungsi mulus dari fitur
        hp = {"epochs": 60, "hidden": 8}
        p1, loss = fit_predict_lstm(X, y, X, seed=1, hp=hp)
        p2, _ = fit_predict_lstm(X, y, X, seed=1, hp=hp)
        self.assertEqual(p1.shape, (300,))
        self.assertTrue(np.all(np.isfinite(p1)) and np.all(p1 >= 0))
        np.testing.assert_allclose(p1, p2)                      # seed sama -> hasil sama
        self.assertGreater(np.corrcoef(np.log1p(p1), np.log1p(y))[0, 1], 0.8)
