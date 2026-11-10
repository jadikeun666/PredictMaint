"""Tes unit modul murni apps/ml_inference/ood_candidates.py (tanpa DB)."""
import copy
import unittest

import numpy as np

from apps.ml_inference import ood_candidates as oc


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for ci, c in enumerate(["HEALTHY", "BALL", "INNER_RACE", "OUTER_RACE"]):
        for gi in range(4 if c == "HEALTHY" else 3):
            base = np.abs(rng.normal(1 + ci, 0.3, (20, 9))) * np.array([0.1, 3, 4, 1, 20, 20, 20, 2, 2000])
            X.append(base)
            y += [c] * 20
            g += ["%s_%d" % (c, gi)] * 20
    return np.vstack(X), np.array(y), np.array(g)


class OodCandidatesTests(unittest.TestCase):
    def test_config_hash_stable_and_sensitive(self):
        h = oc.config_hash()
        self.assertEqual(len(h), 64)
        self.assertEqual(h, oc.config_hash())
        other = copy.deepcopy(oc.CONFIG)
        other["quantiles"] = [0.9, 0.99]
        self.assertNotEqual(h, oc.config_hash(other))

    def test_s9_is_sign_symmetric(self):
        x = np.array([[-3.0, 0.0, 5.0]])
        self.assertTrue(np.allclose(oc.s9(x), -oc.s9(-x)))

    def test_g1_zero_inside_range_positive_outside(self):
        X, y, _ = _data()
        gd = oc.Guard("G1").fit(X, y)
        self.assertEqual(float(gd.score(X).max()), 0.0)
        far = X[:1].copy()
        far[0, 0] = X[:, 0].max() * 3
        self.assertGreater(float(gd.score(far)[0]), 0.0)

    def test_statistical_scores_larger_when_far(self):
        X, y, _ = _data()
        far = oc.apply_shift(X, oc.CONFIG["shifts"]["amp5"])
        for kind in oc.STAT_CANDIDATES:
            gd = oc.Guard(kind).fit(X, y)
            self.assertGreater(float(gd.score(far).mean()), float(gd.score(X).mean()), kind)

    def test_apply_shift_copies_and_scales_right_columns(self):
        X, _, _ = _data()
        before = X.copy()
        out = oc.apply_shift(X, oc.CONFIG["shifts"]["amp2"])
        self.assertTrue(np.array_equal(X, before))
        idx = {f: i for i, f in enumerate(oc.FEATURES)}
        self.assertTrue(np.allclose(out[:, idx["rms"]], X[:, idx["rms"]] * 2))
        self.assertTrue(np.array_equal(out[:, idx["kurtosis"]], X[:, idx["kurtosis"]]))
        imp = oc.apply_shift(X, oc.CONFIG["shifts"]["impulsive"])
        self.assertTrue(np.allclose(imp[:, idx["kurtosis"]], X[:, idx["kurtosis"]] + 10.0))
        self.assertTrue(np.allclose(imp[:, idx["crest_factor"]], X[:, idx["crest_factor"]] * 2.5))

    def test_evaluate_candidate_structure_and_ranges(self):
        X, y, g = _data()
        r = oc.evaluate_candidate(X, y, g, "G2")
        gids = sorted(set(g.tolist()))
        self.assertEqual(r["group_ids"], gids)
        for q in oc.CONFIG["quantiles"]:
            pq = r["per_q"][q]
            self.assertEqual(sorted(pq["frr"]), gids)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in pq["frr"].values()))
            self.assertEqual(set(pq["detection"]), set(oc.CONFIG["shifts"]))
        self.assertGreaterEqual(r["final_tau"][0.99], r["final_tau"][0.95])
        self.assertGreaterEqual(r["per_q"][0.99]["frr_macro"] - 1e-12, 0.0)
        # pada 0,99 FRR tidak boleh lebih tinggi daripada pada 0,95 (tau lebih besar)
        self.assertLessEqual(r["per_q"][0.99]["frr_macro"], r["per_q"][0.95]["frr_macro"] + 1e-12)

    def test_evaluate_needs_three_groups(self):
        X, y, g = _data()
        two = np.isin(g, sorted(set(g.tolist()))[:2])
        with self.assertRaises(ValueError):
            oc.evaluate_candidate(X[two], y[two], g[two], "G2")

    def _fake(self, frr, det):
        q = oc.CONFIG["selection_q"]
        return {"per_q": {q: {"frr_macro": frr, "detection_mean": det}}}

    def test_select_candidate_rules(self):
        # semua tidak eligible
        res = {k: self._fake(0.5, 0.9) for k in oc.STAT_CANDIDATES}
        self.assertIsNone(oc.select_candidate(res)[0])
        # G4 terbaik jauh -> G4
        res = {"G2": self._fake(0.05, 0.50), "G3": self._fake(0.05, 0.60), "G4": self._fake(0.05, 0.90)}
        self.assertEqual(oc.select_candidate(res)[0], "G4")
        # dalam toleransi -> yang paling sederhana
        res = {"G2": self._fake(0.05, 0.89), "G3": self._fake(0.05, 0.90), "G4": self._fake(0.05, 0.90)}
        self.assertEqual(oc.select_candidate(res)[0], "G2")
        # yang melebihi batas FRR tidak boleh dipilih walau deteksi tertinggi
        res = {"G2": self._fake(0.05, 0.50), "G3": self._fake(0.30, 0.99), "G4": self._fake(0.05, 0.40)}
        self.assertEqual(oc.select_candidate(res)[0], "G2")

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            oc.Guard("GX")


if __name__ == "__main__":
    unittest.main()
