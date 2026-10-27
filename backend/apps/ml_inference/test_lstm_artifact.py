"""
apps/ml_inference/test_lstm_artifact.py — artefak LSTM + pembangun entri registry (Giliran 3).
Di-skip otomatis bila torch tidak tersedia.
"""
import json
import os
import tempfile
import unittest

import numpy as np
from django.test import SimpleTestCase

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from apps.ml_inference import lstm_rul as lr
    from apps.ml_inference import model_registry
    from apps.ml_inference import rul_eval as ev
    from apps.ml_inference.feature_extraction import FEATURE_ORDER
    from apps.ml_inference.management.commands.register_lstm_rul import build_registry_entry


def _fake_eval():
    def methods(base):
        return {m: {"mae": base + i, "rmse": base + 10 + i, "phm": 1.0 + i, "mae_last10pct": 5.0 + i,
                    "bias": -3.0 + i, "mae_seed_std": 1.0}
                for i, m in enumerate(("const", "hi_extrap", "ridge", "lstm"))}
    rows = [{"fold": "a", "test_traj": "T1", "traj_hours": 800.0, "methods": methods(100.0)},
            {"fold": "b", "test_traj": "T2", "traj_hours": 160.0, "methods": methods(120.0)}]
    agg_m = {m: {k: {"mean": 1.5, "std": 0.5} for k in ("mae", "rmse", "phm", "mae_last10pct", "bias")}
             for m in ("const", "hi_extrap", "ridge", "lstm")}
    agg = {"n_test_trajectories": 2, "methods": agg_m,
           "lstm_beats_all_baselines_mae": 1, "lstm_beats_all_baselines_phm": 0}
    return {"config": {"hp": {}, "seeds": [11, 22, 33]}, "training_dataset_version_hash": "abc123",
            "schemes": {"loto": {"rows": rows, "aggregate": agg},
                        "loto_excl_set3": {"rows": rows, "aggregate": agg}}}


@unittest.skipUnless(HAS_TORCH, "torch tidak tersedia")
class ArtifactTests(SimpleTestCase):
    def _fit(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 6, 9)).astype(np.float32)
        y = np.exp(2.0 + 0.5 * X[:, -1, 0])
        model, ts, _ = lr.fit_lstm(X, y, seed=3, hp={"hidden": 8, "epochs": 3})
        return X, y, model, ts

    def test_fit_predict_wrapper_matches_split_api(self):
        X, y, model, ts = self._fit()
        p_wrapper, _ = lr.fit_predict_lstm(X, y, X, seed=3, hp={"hidden": 8, "epochs": 3})
        np.testing.assert_allclose(p_wrapper, lr.predict_lstm(model, ts, X))

    def test_save_load_roundtrip_identical(self):
        X, y, model, ts = self._fit()
        mean, std = np.arange(9.0), np.arange(9.0) + 1
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v", "model.pt")
            lr.save_artifact(path, model, ts, mean, std, FEATURE_ORDER, 6, {"hidden": 8}, meta={"k": "v"})
            m2, t2, mean2, std2, info = lr.load_artifact(path)
        np.testing.assert_allclose(lr.predict_lstm(model, ts, X), lr.predict_lstm(m2, t2, X), atol=1e-9)
        np.testing.assert_array_equal(mean2, mean)
        np.testing.assert_array_equal(std2, std)
        self.assertEqual(info["feature_order"], list(FEATURE_ORDER))
        self.assertEqual(info["seq_len"], 6)
        self.assertEqual(info["meta"], {"k": "v"})


@unittest.skipUnless(HAS_TORCH, "torch tidak tersedia")
class RegistryEntryTests(SimpleTestCase):
    def test_entry_has_required_fields_and_values_come_from_eval(self):
        sc = ev.Scaler(np.zeros(9), np.ones(9))
        e = build_registry_entry(
            _fake_eval(), model_version="lstm-rul-ims-vX", artifact_path="/app/ml_artifacts/x/model.pt",
            scaler=sc, tstats={"mu": 5.0, "sd": 1.0}, hp={"hidden": 32}, final_seed=11,
            n_train_windows=100, n_train_files=50, n_train_trajs=4,
            trained_at="2026-10-27T00:00:00+00:00", eval_date="2026-10-27")
        for f in model_registry.REQUIRED_FIELDS:
            self.assertIn(f, e)
        self.assertEqual(e["model_type"], "RUL")
        self.assertEqual(e["training_dataset"], "IMS")
        self.assertIsNone(e["eval_accuracy"])
        self.assertEqual(e["eval_mae_hours"], 1.5)               # dari aggregate, bukan diketik
        self.assertEqual(e["training_dataset_version_hash"], "abc123")
        self.assertEqual(e["rul_confidence_band_hours"], e["eval_rmse_hours"])
        self.assertEqual(len(e["cv_fold_details"]), 2)
        self.assertTrue(any("160.0" in n for n in e["evaluation_notes"]))   # umur lintasan terpendek disebut
        json.dumps(e)                                            # serializable
