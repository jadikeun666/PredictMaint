"""Tes Sesi D G5: filter NOT_DEPLOYED, update_entry_fields, builder entri CNN/RF-PU, ringkasan keyakinan."""
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from django.test import SimpleTestCase

from apps.ml_inference import model_registry as mr
from apps.ml_inference.management.commands import register_sesi_d as rs


def _entry(version, date, mtype="CLASSIFICATION", **extra):
    e = {"model_version": version, "model_type": mtype, "training_dataset": "CWRU",
         "training_dataset_version_hash": "d", "eval_date": date, "artifact_path": "/x/" + version}
    e.update(extra)
    return e


def _with_registry(entries):
    d = tempfile.TemporaryDirectory()
    p = Path(d.name) / "model_registry.json"
    p.write_text(json.dumps(entries))
    return d, mock.patch.object(mr, "REGISTRY_PATH", p)


def _subset(acc, maj):
    return {"n": 10, "acc": acc, "acc_thr": acc, "macro_f1": 0.2, "mrecall": 0.3,
            "recall_argmax": [0.0, 0.1, 0.5], "majority": maj}


def _seed(acc):
    return {"SEMUA": _subset(acc, 0.42), "BUATAN": _subset(acc, 0.35), "NYATA": _subset(acc, 0.40)}


G4 = {"config": {"epochs": 30, "input": "x"}, "config_hash": "h", "dataset_hash": "d", "smoke": False,
      "cv": {"accuracy": 0.99, "accuracy_std": 0.01, "macro_f1": 0.98, "folds": [{"fold": 1}],
             "confusion": [[1]], "folds_identical_to_rf": True},
      "pu": {"seed42": _seed(0.3), "seed43": _seed(0.2), "diag_3class_seed42": {"BUATAN": _subset(0.2, 0.35), "NYATA": _subset(0.1, 0.4)}}}
G4B = {"config_hash": "h", "folds": [
    {"instance": "B007", "class": "BALL", "n": 35, "cnn_recall": 0.0, "rf_recall": 0.5, "cnn_pred": {}, "rf_pred": {}},
    {"instance": "IR007", "class": "INNER_RACE", "n": 35, "cnn_recall": 1.0, "rf_recall": 1.0, "cnn_pred": {}, "rf_pred": {}}]}
RF = _entry("rf-baseline-cwru-v1.1", "2026-09-13", eval_accuracy=0.86)


class RegistryDeploymentTests(SimpleTestCase):
    def test_get_latest_skips_not_deployed_even_if_newer(self):
        d, patch = _with_registry([_entry("rf", "2026-09-13"), _entry("cnn", "2026-11-06", deployment="NOT_DEPLOYED")])
        with d, patch:
            self.assertEqual(mr.get_latest_entry("CLASSIFICATION")["model_version"], "rf")

    def test_entry_without_field_is_deployable(self):
        d, patch = _with_registry([_entry("old", "2026-09-11"), _entry("new", "2026-09-13")])
        with d, patch:
            self.assertEqual(mr.get_latest_entry("CLASSIFICATION")["model_version"], "new")

    def test_other_type_unaffected(self):
        d, patch = _with_registry([_entry("lstm", "2026-10-27", mtype="RUL"), _entry("cnn", "2026-11-06", deployment="NOT_DEPLOYED")])
        with d, patch:
            self.assertEqual(mr.get_latest_entry("RUL")["model_version"], "lstm")

    def test_update_fills_null_and_refuses_overwrite(self):
        d, patch = _with_registry([_entry("rf", "2026-09-13", cross_dataset_pu_accuracy=None)])
        with d, patch:
            mr.update_entry_fields("rf", {"cross_dataset_pu_accuracy": 0.18, "cross_dataset_pu_note": "n"})
            self.assertEqual(mr.load_registry()[0]["cross_dataset_pu_accuracy"], 0.18)
            with self.assertRaises(ValueError):
                mr.update_entry_fields("rf", {"cross_dataset_pu_accuracy": 0.5})
            self.assertEqual(mr.load_registry()[0]["cross_dataset_pu_accuracy"], 0.18)

    def test_update_refuses_ambiguous_or_missing_version(self):
        d, patch = _with_registry([_entry("rf", "2026-09-11"), _entry("rf", "2026-09-11")])
        with d, patch:
            with self.assertRaises(ValueError):
                mr.update_entry_fields("rf", {"x": 1})
            with self.assertRaises(ValueError):
                mr.update_entry_fields("tidak-ada", {"x": 1})


class BuilderTests(SimpleTestCase):
    def test_cnn_entry_fields_and_numbers_come_from_inputs(self):
        e = rs.build_cnn_entry(G4, G4B, RF, artifact_path="/a/model.pt", trained_at="t", eval_date="2026-11-06")
        for f in mr.REQUIRED_FIELDS:
            self.assertIn(f, e)
        self.assertEqual(e["deployment"], "NOT_DEPLOYED")
        self.assertEqual(e["eval_accuracy"], 0.99)
        self.assertAlmostEqual(e["eval_leave_instance_out"]["cnn_accuracy"], 0.5)
        self.assertAlmostEqual(e["eval_leave_instance_out"]["rf_accuracy"], 0.75)
        self.assertAlmostEqual(e["cross_dataset_pu_accuracy"], 0.25)
        self.assertEqual(e["training_dataset_version_hash"], "d")
        self.assertTrue(any("di bawah kelas mayoritas" in n for n in e["evaluation_notes"]))

    def test_rf_pu_fields_headline_is_real_subset(self):
        pu = {"subsets": {"SEMUA": _subset(0.26, 0.42), "BUATAN (+healthy)": _subset(0.27, 0.35),
                          "NYATA (+healthy)": _subset(0.18, 0.40)}}
        f = rs.build_rf_pu_fields(pu, "src.json")
        self.assertAlmostEqual(f["cross_dataset_pu_accuracy"], 0.18)
        self.assertIn("di bawah", f["cross_dataset_pu_note"])

    def test_confidence_summary(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            probs = np.array([[0, 0, 1.0, 0], [0, 0, 0.9, 0.1], [0, 0.5, 0.2, 0.3], [0, 0, 0.95, 0.05]], dtype=np.float32)
            np.savez(d / "p.npz", probs=probs, classes=np.array(["BALL", "HEALTHY", "INNER_RACE", "OUTER_RACE"]))
            np.savez(d / "i.npz", label=np.array(["HEALTHY", "INNER_RACE", "OUTER_RACE", "OUTER_RACE", "X"]),
                     rpm_sane=np.array([True, True, True, True, False]))
            s = rs.confidence_summary(d / "p.npz", d / "i.npz")
        self.assertEqual(s["pred_INNER_RACE_conf_ge_085"]["n"], 3)
        self.assertAlmostEqual(s["pred_INNER_RACE_conf_ge_085"]["precision"], 1 / 3)
        self.assertAlmostEqual(s["per_true_label"]["OUTER_RACE"]["conf_lt_060"], 0.5)
