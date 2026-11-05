"""
apps/ml_inference/test_pu_dataset.py — pemuat PU, fitur generik, cache evaluasi (Sesi D, Giliran 2).
Test yang butuh apps.signal_processing asli atau pu_bearing_labels.json di-skip otomatis bila tidak ada.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from django.test import SimpleTestCase
from scipy.io import savemat

from apps.ml_inference import pu_cache
from apps.ml_inference import pu_dataset as pu
from apps.ml_inference.feature_extraction import FEATURE_ORDER

try:
    from apps.signal_processing import dsp_features as _dsp
    HAVE_SIGNAL_PROCESSING = callable(getattr(_dsp, "remove_dc", None))
except ImportError:
    HAVE_SIGNAL_PROCESSING = False

GEOM = {"num_balls": 8, "ball_diameter_mm": 6.75, "pitch_diameter_mm": 29.05, "contact_angle_deg": 0.0}


def write_pu_file(path, vib, speed_value=899.7, speed_len=200):
    """File .mat sintetis dengan struktur seperti PU: top.Y = array struct (Name, Data)."""
    Y = np.zeros((3,), dtype=[("Name", "O"), ("Data", "O")])
    Y[0] = ("force", np.zeros(50))
    Y[1] = ("speed", np.full(speed_len, speed_value))
    Y[2] = ("vibration_1", np.asarray(vib, dtype=float))
    stem = Path(path).stem
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), {stem: {"Y": Y, "X": np.zeros(1)}})


class FileNameTests(SimpleTestCase):
    def test_parse(self):
        i = pu.parse_file_name("N15_M07_F04_KB24_13.mat")
        self.assertEqual(i, {"speed_code": "15", "torque_code": "07", "force_code": "04", "code": "KB24", "index": 13})

    def test_parse_rejects(self):
        for bad in ("K001.pdf", "N15_M07_F04_K001.mat", "N15_M07_F04_XX01_1.mat"):
            with self.assertRaises(ValueError):
                pu.parse_file_name(bad)

    def test_list_files_sorted_numerically(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("N09_M07_F10_K001_10.mat", "N09_M07_F10_K001_2.mat", "N09_M07_F10_K001_1.mat", "N15_M01_F10_K001_1.mat"):
                write_pu_file(Path(d) / "K001" / n, np.zeros(10))
            names = [p.name for p in pu.list_files(d, "K001")]
            self.assertEqual(names, ["N09_M07_F10_K001_1.mat", "N09_M07_F10_K001_2.mat", "N09_M07_F10_K001_10.mat",
                                     "N15_M01_F10_K001_1.mat"])
            with self.assertRaises(FileNotFoundError):
                pu.list_files(d, "KA01")


class SignalTests(SimpleTestCase):
    def test_cut_windows(self):
        self.assertEqual(pu.cut_windows(np.arange(255900.0)).shape, (3, 64000))     # < 4,0 detik -> 3 jendela penuh
        w = pu.cut_windows(np.arange(256823.0))
        self.assertEqual(w.shape, (4, 64000))
        self.assertEqual(w[1, 0], 64000.0)                       # jendela berurutan, tanpa overlap
        with self.assertRaises(ValueError):
            pu.cut_windows(np.zeros(1000))

    def test_to_cnn_window_shape_dtype_and_tone(self):
        t = np.arange(64000) / 64000.0
        w = 2.0 * np.sin(2 * np.pi * 100.0 * t)                  # 100 Hz, amplitudo 2
        r = pu.to_cnn_window(w)
        self.assertEqual(r.shape, (12000,))
        self.assertEqual(r.dtype, np.float32)
        spec = np.abs(np.fft.rfft(r)) / (len(r) / 2)
        self.assertEqual(int(np.argmax(spec)), 100)              # 1,0 detik -> resolusi 1 Hz
        self.assertAlmostEqual(float(spec[100]), 2.0, places=2)  # amplitudo terjaga

    def test_rpm_sanity(self):
        self.assertTrue(pu.rpm_sanity(899.7, "09"))
        self.assertTrue(pu.rpm_sanity(1499.6, "15"))
        self.assertFalse(pu.rpm_sanity(1200.0, "15"))


class ReadFileTests(SimpleTestCase):
    def test_read_pu_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "K001" / "N09_M07_F10_K001_1.mat"
            vib = np.random.default_rng(0).normal(size=256000)
            write_pu_file(p, vib, speed_value=899.7)
            x, rpm = pu.read_pu_file(p)
        np.testing.assert_allclose(x, vib)
        self.assertAlmostEqual(rpm, 899.7)

    def test_missing_channel_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "N09_M07_F10_K001_1.mat"
            Y = np.zeros((1,), dtype=[("Name", "O"), ("Data", "O")])
            Y[0] = ("force", np.zeros(5))
            savemat(str(p), {"N09_M07_F10_K001_1": {"Y": Y}})
            with self.assertRaises(ValueError):
                pu.read_pu_file(p)


@unittest.skipUnless(pu.LABELS_PATH.exists(), "pu_bearing_labels.json belum ada")
class LabelsFileTests(SimpleTestCase):
    def test_counts_match_documented_table(self):
        labels = pu.load_labels()
        self.assertEqual(len(labels), 32)
        from collections import Counter
        c = Counter((e["label"], e["origin"]) for e in labels.values())
        self.assertEqual(c[("HEALTHY", "healthy")], 6)
        self.assertEqual((c[("OUTER_RACE", "artificial")], c[("OUTER_RACE", "real")]), (6, 5))
        self.assertEqual((c[("INNER_RACE", "artificial")], c[("INNER_RACE", "real")]), (5, 4))
        self.assertEqual(c[("MIXED", "artificial")] + c[("MIXED", "real")], 6)
        self.assertEqual(len(pu.eval_codes(labels)), 26)
        self.assertNotIn("KA08", pu.eval_codes(labels))          # komponen 'AR' belum dipahami -> dikeluarkan

    def test_two_geometries_are_kept_per_code(self):
        labels = pu.load_labels()
        self.assertEqual(labels["KA01"]["geometry"]["pitch_diameter_mm"], 29.05)
        self.assertEqual(labels["KA04"]["geometry"]["pitch_diameter_mm"], 28.55)


class CacheBuildTests(SimpleTestCase):
    def _fake(self):
        def fake_extract(w, fs, rpm, geometry):
            return {k: float(np.std(w)) + i for i, k in enumerate(FEATURE_ORDER)}
        return fake_extract

    def test_build_cache_with_variable_windows_per_file(self):
        from apps.ml_inference import features_generic as fg
        labels = {"K001": {"label": "HEALTHY", "origin": "healthy", "geometry": GEOM},
                  "KA01": {"label": "OUTER_RACE", "origin": "artificial", "geometry": GEOM}}
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            for code in ("K001", "KA01"):
                for n, spd in (("N09_M07_F10_%s_1.mat", 899.7), ("N15_M07_F10_%s_1.mat", 1499.6)):
                    write_pu_file(Path(root) / code / (n % code), rng.normal(size=256100), speed_value=spd)
            # file lebih pendek dari 4,0 detik (255.000 sampel -> hanya 3 jendela penuh) + RPM tidak sesuai kode N15
            write_pu_file(Path(root) / "KA01" / "N15_M01_F10_KA01_1.mat", rng.normal(size=255000), speed_value=1200.0)
            with mock.patch.object(fg, "extract_features", self._fake()):
                st = pu_cache.build_cache(root, out, labels, workers=2)
            idx = np.load(os.path.join(out, "index.npz"))
            raw = np.load(os.path.join(out, "raw12k.npy"), mmap_mode="r")
            self.assertEqual(st["n_files"], 5)
            self.assertEqual(st["n_windows"], 19)                       # 4+4 (K001) + 4+4+3 (KA01)
            self.assertEqual(st["windows_per_file"], {3: 1, 4: 4})
            self.assertEqual(raw.shape, (19, 12000))
            self.assertEqual(raw.dtype, np.float32)
            self.assertEqual(idx["features"].shape, (19, 9))
            self.assertEqual(len(idx["label"]), 19)
            self.assertEqual(int((idx["label"] == "HEALTHY").sum()), 8)
            self.assertEqual(sorted(set(idx["window_index"].tolist())), [0, 1, 2, 3])
            self.assertEqual(st["n_rpm_insane"], 3)                     # 3 jendela dari file RPM 1200 pada kode N15
            self.assertTrue(np.all(np.isfinite(idx["features"])))
            self.assertTrue(np.all(np.abs(raw[:]) > 0))                 # tidak ada baris sisa prealokasi
            for tmp in ("raw12k.tmp.npy", "raw12k.final.tmp.npy", "index.tmp.npz"):
                self.assertFalse(os.path.exists(os.path.join(out, tmp)), tmp)

    def test_failure_leaves_no_temp_files(self):
        from apps.ml_inference import features_generic as fg
        labels = {"K001": {"label": "HEALTHY", "origin": "healthy", "geometry": GEOM}}
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            write_pu_file(Path(root) / "K001" / "N09_M07_F10_K001_1.mat", np.zeros(1000), speed_value=899.7)   # < 1 jendela
            with mock.patch.object(fg, "extract_features", self._fake()):
                with self.assertRaises(ValueError):
                    pu_cache.build_cache(root, out, labels, workers=1)
            self.assertEqual([p for p in os.listdir(out)], [])          # tidak ada sisa berkas sementara


@unittest.skipUnless(HAVE_SIGNAL_PROCESSING, "apps.signal_processing asli tidak tersedia")
class FeatureParityTests(SimpleTestCase):
    def test_generic_equals_ims_function(self):
        from apps.ml_inference import features_generic as fg
        from apps.ml_inference import ims_dataset as ims
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.1, ims.IMS_SNAPSHOT_SAMPLES)
        a = fg.extract_features(x, ims.IMS_SAMPLE_RATE_HZ, ims.IMS_RPM, ims.IMS_GEOMETRY)
        b = ims.extract_features_ims(x)
        for k in FEATURE_ORDER:
            self.assertAlmostEqual(a[k], b[k], places=9, msg=k)

    def test_pu_geometry_gives_finite_features(self):
        from apps.ml_inference import features_generic as fg
        rng = np.random.default_rng(1)
        f = fg.extract_features(rng.normal(0, 0.4, 64000), pu.PU_SAMPLE_RATE_HZ, 1500.0, GEOM)
        self.assertEqual(list(f), list(FEATURE_ORDER))
        self.assertTrue(all(np.isfinite(v) for v in f.values()))
