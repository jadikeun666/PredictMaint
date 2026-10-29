"""apps/ml_inference/test_rul_alert_policy.py — simulasi kebijakan alert RUL (numpy murni)."""
import numpy as np
from django.test import SimpleTestCase

from apps.ml_inference import rul_alert_policy as pol

TH = pol.Thresholds(critical=168.0, high=336.0, warning=720.0)


def _case(true, pred, hours=None, traj="T", seed=1):
    true = np.asarray(true, dtype=float)
    return pol.Case(traj, seed, true, np.asarray(pred, dtype=float), float(hours if hours is not None else true[0]))


class PrimitiveTests(SimpleTestCase):
    def test_smooth_is_causal_median(self):
        x = np.array([10.0, 0.0, 10.0, 10.0, 10.0])
        np.testing.assert_allclose(pol.smooth_causal(x, 1), x)
        s = pol.smooth_causal(x, 3)
        np.testing.assert_allclose(s, [10.0, 5.0, 10.0, 10.0, 10.0])   # tepi awal memakai jendela pendek
        y = x.copy(); y[-1] = 999.0
        np.testing.assert_allclose(pol.smooth_causal(y, 3)[:-1], s[:-1])  # nilai masa depan tidak memengaruhi

    def test_severity_rank_inclusive_thresholds(self):
        r = pol.severity_rank([100, 168, 168.01, 336, 500, 720, 720.01], TH)
        self.assertEqual(r.tolist(), [3, 3, 2, 2, 1, 1, 0])

    def test_persistence_requires_m_consecutive(self):
        pred = np.array([1000, 100, 1000, 100, 100, 100, 100], dtype=float)   # CRITICAL pada idx 1,3,4,5,6
        conf = pol.confirmed_levels(pred, pol.Policy(1, 3, False), TH)
        self.assertEqual(pol.first_alert_index(conf, 3), 5)                   # baru 3 berturut-turut di idx 3,4,5

    def test_trend_blocks_non_decreasing(self):
        flat = np.full(40, 100.0)
        self.assertIsNone(pol.first_alert_index(pol.confirmed_levels(flat, pol.Policy(1, 1, True), TH), 3))
        falling = np.linspace(400, 50, 40)
        self.assertIsNotNone(pol.first_alert_index(pol.confirmed_levels(falling, pol.Policy(1, 1, True), TH), 3))


class OutcomeTests(SimpleTestCase):
    def test_outcomes(self):
        true = np.linspace(1000, 0, 101)             # RUL asli turun 10 jam/jendela
        f = lambda idx, h=168.0, lm=48.0: pol.classify_outcome(true, idx, h, lm, 2.0, 1000.0)
        self.assertEqual(f(None), "MISS")
        self.assertEqual(f(10), "PREMATURE")         # RUL asli 900 > 2*168
        self.assertEqual(f(85), "HIT")               # RUL asli 150
        self.assertEqual(f(98), "LATE")              # RUL asli 20 < 48
        self.assertEqual(pol.classify_outcome(true, 5, 336.0, 168.0, 2.0, 100.0), "NA")   # umur < lead_min


class EvaluateTests(SimpleTestCase):
    def test_perfect_predictor_hits_everywhere(self):
        true = np.linspace(1000, 0, 200)
        cases = [_case(true, true, hours=1000.0, traj=f"T{i}", seed=i) for i in range(3)]
        res = pol.evaluate_policy(cases, pol.BASELINE_POLICY, TH)
        self.assertEqual(res["levels"]["CRITICAL"]["HIT"], 3)
        self.assertEqual(res["levels"]["HIGH"]["HIT"], 3)
        self.assertEqual(res["levels"]["CRITICAL"]["premature_rate"], 0.0)

    def test_early_biased_predictor_is_premature_and_smoothing_does_not_fix_bias(self):
        true = np.linspace(1000, 0, 200)
        res = pol.evaluate_policy([_case(true, true - 400.0, hours=1000.0)], pol.Policy(12, 6, False), TH)
        self.assertEqual(res["levels"]["CRITICAL"]["PREMATURE"], 1)

    def test_short_trajectory_is_na_for_high(self):
        true = np.linspace(160, 0, 100)
        res = pol.evaluate_policy([_case(true, true, hours=160.0)], pol.BASELINE_POLICY, TH)
        self.assertEqual(res["levels"]["HIGH"]["NA"], 1)
        self.assertEqual(res["levels"]["HIGH"]["n_eval"], 0)
        self.assertIsNone(res["levels"]["HIGH"]["hit_rate"])
        self.assertEqual(res["levels"]["CRITICAL"]["HIT"], 1)


class SelectionTests(SimpleTestCase):
    def test_grid_size_and_unique_names(self):
        g = pol.build_grid()
        self.assertEqual(len(g), 24)
        self.assertEqual(len({p.name for p in g}), 24)

    def test_none_when_noisy_predictor(self):
        rng = np.random.default_rng(0)
        true = np.linspace(1000, 0, 300)
        cases = [_case(true, true - 450 + rng.normal(0, 80, 300), hours=1000.0, seed=s) for s in range(3)]
        results = [pol.evaluate_policy(cases, p, TH) for p in pol.build_grid()]
        name, why = pol.select_policy(results)
        self.assertIsNone(name)
        self.assertIn("tidak ada", why)

    def test_selects_simplest_among_equal(self):
        true = np.linspace(1000, 0, 300)
        cases = [_case(true, true, hours=1000.0, seed=s) for s in range(3)]
        results = [pol.evaluate_policy(cases, p, TH) for p in pol.build_grid()]
        name, _ = pol.select_policy(results)
        self.assertEqual(name, "K1-M1-notrend")      # semua sempurna -> yang paling sederhana
