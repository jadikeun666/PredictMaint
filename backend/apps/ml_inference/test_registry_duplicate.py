"""
Regresi Sesi F: registry menolak model_version duplikat (sebelumnya append_entry hanya memeriksa
field wajib, sehingga melatih ulang versi yang sama menambah entri ganda dan membuat
update_entry_fields menolak versi itu sebagai ambigu). REGISTRY_PATH dialihkan ke berkas sementara.
"""
import json
import tempfile
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

from apps.ml_inference import model_registry


def _entry(version, **extra):
    e = {
        "model_version": version, "model_type": "CLASSIFICATION", "training_dataset": "CWRU",
        "training_dataset_version_hash": "h", "eval_date": "2026-09-11", "artifact_path": "/x/model.joblib",
    }
    e.update(extra)
    return e


class RegistryDuplicateTests(SimpleTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "model_registry.json"
        p = mock.patch.object(model_registry, "REGISTRY_PATH", self.path)
        p.start()
        self.addCleanup(p.stop)

    def _raw(self):
        return self.path.read_text() if self.path.exists() else None

    def test_append_duplicate_version_raises_and_leaves_file_unchanged(self):
        model_registry.append_entry(_entry("m-v1.0"))
        before = self._raw()
        with self.assertRaises(ValueError):
            model_registry.append_entry(_entry("m-v1.0", trained_at="lain"))
        self.assertEqual(self._raw(), before)
        self.assertEqual(len(model_registry.load_registry()), 1)

    def test_append_new_version_still_works(self):
        model_registry.append_entry(_entry("m-v1.0"))
        model_registry.append_entry(_entry("m-v1.1"))
        self.assertEqual([e["model_version"] for e in model_registry.load_registry()], ["m-v1.0", "m-v1.1"])

    def test_has_version(self):
        self.assertFalse(model_registry.has_version("m-v1.0"))
        model_registry.append_entry(_entry("m-v1.0"))
        self.assertTrue(model_registry.has_version("m-v1.0"))
        self.assertFalse(model_registry.has_version("m-v9.9"))

    def test_update_entry_fields_unchanged_fills_only_empty(self):
        model_registry.append_entry(_entry("m-v1.0", cross_dataset_pu_accuracy=None))
        model_registry.update_entry_fields("m-v1.0", {"cross_dataset_pu_accuracy": 0.18})
        self.assertEqual(model_registry.load_registry()[0]["cross_dataset_pu_accuracy"], 0.18)
        with self.assertRaises(ValueError):
            model_registry.update_entry_fields("m-v1.0", {"cross_dataset_pu_accuracy": 0.99})
