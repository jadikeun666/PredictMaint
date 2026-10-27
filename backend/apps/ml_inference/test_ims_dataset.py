"""
apps/ml_inference/test_ims_dataset.py — test ims_dataset (Giliran 1, Sesi B).
Tes data nyata di-skip otomatis kalau dataset IMS belum ter-mount.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from django.test import SimpleTestCase

from apps.ml_inference import ims_dataset as ims
from apps.ml_inference.feature_extraction import FEATURE_ORDER
from apps.signal_processing.bearing_frequencies import compute_fault_frequencies

FAKE = ims.Trajectory(
    traj_id="T-FAKE", set_no=9, folder="fake", bearing_no=1, failure_mode="x",
    columns=(0,), n_columns_in_file=1, readme_n_files=3,
    readme_start="2004.01.01.00.00.00", readme_end="2004.01.01.00.20.00",
)


class TimestampTests(SimpleTestCase):
    def test_parse_and_delta(self):
        a = ims.parse_snapshot_timestamp("2004.02.12.10.32.39")
        b = ims.parse_snapshot_timestamp("2004.02.12.10.42.39")
        self.assertEqual(b - a, 600.0)

    def test_parse_rejects_bad_name(self):
        for bad in ("notes.txt", "2004.02.12", "txt"):
            with self.assertRaises(ValueError):
                ims.parse_snapshot_timestamp(bad)


class LabelTests(SimpleTestCase):
    def test_rul_from_timeline(self):
        np.testing.assert_allclose(ims.compute_rul_hours([0, 3600, 7200]), [2.0, 1.0, 0.0])

    def test_rul_requires_sorted(self):
        with self.assertRaises(ValueError):
            ims.compute_rul_hours([10, 5])

    def test_rul_with_separate_failure_ts(self):
        # snapshot terakhir dibuang (sinyal mati), kegagalan tetap di t=7200
        np.testing.assert_allclose(ims.compute_rul_hours([0, 3600], failure_ts=7200), [2.0, 1.0])

    def test_rul_failure_ts_before_last_snapshot_raises(self):
        with self.assertRaises(ValueError):
            ims.compute_rul_hours([0, 3600], failure_ts=1800)

    def test_hi_range_and_monotone(self):
        rms = np.linspace(1, 5, 50)
        kurt = np.linspace(3, 9, 50)
        hi = ims.compute_health_indicator(rms, kurt)
        self.assertAlmostEqual(hi.min(), 0.0)
        self.assertAlmostEqual(hi.max(), 1.0)
        self.assertTrue(np.all(np.diff(hi) >= 0))

    def test_hi_constant_series_is_zero(self):
        hi = ims.compute_health_indicator(np.ones(10), np.ones(10))
        np.testing.assert_array_equal(hi, np.zeros(10))


class DiscoverTests(SimpleTestCase):
    def test_filter_sort_and_counts(self):
        with tempfile.TemporaryDirectory() as root:
            d = Path(root) / "fake"
            d.mkdir()
            for n in ("2004.01.01.00.10.00", "2004.01.01.00.00.00",
                      "2004.01.01.00.20.00", "2004.01.01.00.30.00", "notes.txt"):
                (d / n).write_text("0\n")
            lst = ims.discover_snapshots(FAKE, root)
            self.assertEqual([s.name for s in lst.entries],
                             ["2004.01.01.00.00.00", "2004.01.01.00.10.00", "2004.01.01.00.20.00"])
            self.assertEqual(lst.n_after_readme_end, 1)
            self.assertEqual(lst.n_ignored_non_snapshot, 1)
            self.assertEqual(lst.n_in_folder, 5)

    def test_use_all_files_keeps_files_after_readme_end(self):
        import dataclasses
        t = dataclasses.replace(FAKE, use_all_files=True)
        with tempfile.TemporaryDirectory() as root:
            d = Path(root) / "fake"
            d.mkdir()
            for n in ("2004.01.01.00.00.00", "2004.01.01.00.20.00", "2004.01.01.00.30.00"):
                (d / n).write_text("0\n")
            lst = ims.discover_snapshots(t, root)
            self.assertEqual(len(lst.entries), 3)
            self.assertEqual(lst.n_after_readme_end, 1)
            self.assertTrue(lst.after_end_used)

    def test_s3_trajectory_uses_all_files_and_others_do_not(self):
        flags = {t.traj_id: t.use_all_files for t in ims.TRAJECTORIES}
        self.assertEqual(flags, {"IMS-S1-B3": False, "IMS-S1-B4": False,
                                 "IMS-S2-B1": False, "IMS-S3-B3": True})

    def test_missing_folder_raises(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                ims.discover_snapshots(FAKE, root)

    def test_hash_changes_with_file_list(self):
        a = [ims.Snapshot("2004.01.01.00.00.00", Path("x"), 0.0, 10)]
        b = [ims.Snapshot("2004.01.01.00.00.00", Path("x"), 0.0, 11)]
        self.assertNotEqual(ims.file_list_hash(a), ims.file_list_hash(b))


class LoadSnapshotTests(SimpleTestCase):
    def test_shape_validation(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "2004.01.01.00.00.00"
            np.savetxt(p, np.arange(8.0).reshape(4, 2), delimiter="\t")
            arr = ims.load_snapshot(p, n_columns=2, expected_rows=4)
            self.assertEqual(arr.shape, (4, 2))
            with self.assertRaises(ValueError):
                ims.load_snapshot(p, n_columns=2, expected_rows=5)


class DeadSignalTests(SimpleTestCase):
    def test_dead_vs_normal(self):
        rng = np.random.default_rng(1)
        dead = rng.normal(0, 0.001, size=(500, 4))
        normal = rng.normal(0, 0.1, size=(500, 4))
        self.assertTrue(ims.is_dead_snapshot(dead, (0,)))
        self.assertFalse(ims.is_dead_snapshot(normal, (0,)))

    def test_only_trajectory_columns_are_checked(self):
        rng = np.random.default_rng(2)
        x = np.stack([rng.normal(0, 0.001, 500), rng.normal(0, 0.1, 500)], axis=1)
        self.assertTrue(ims.is_dead_snapshot(x, (0,)))
        self.assertFalse(ims.is_dead_snapshot(x, (1,)))
        self.assertFalse(ims.is_dead_snapshot(x, (0, 1)))      # semua kanal harus mati


class BuildTrajectoryTests(SimpleTestCase):
    def test_dead_last_snapshot_dropped_but_failure_time_kept(self):
        t = ims.Trajectory(
            traj_id="T-BUILD", set_no=9, folder="b", bearing_no=1, failure_mode="x",
            columns=(0,), n_columns_in_file=1, readme_n_files=6,
            readme_start="2004.01.01.00.00.00", readme_end="2004.01.01.00.50.00",
        )
        rng = np.random.default_rng(3)
        names = [f"2004.01.01.00.{m:02d}.00" for m in (0, 10, 20, 30, 40, 50)]
        with tempfile.TemporaryDirectory() as root:
            d = Path(root) / "b"
            d.mkdir()
            for i, n in enumerate(names):
                sd = 0.001 if i == 5 else 0.1 * (1 + i)          # file terakhir mati
                np.savetxt(d / n, rng.normal(0, sd, size=ims.IMS_SNAPSHOT_SAMPLES))
            lst = ims.discover_snapshots(t, root)
            data = ims.build_trajectory(t, lst, workers=2)
        self.assertEqual(data.meta["n_files"], 6)
        self.assertEqual(data.meta["n_valid"], 5)
        self.assertEqual(data.meta["n_dead_excluded"], 1)
        self.assertEqual(data.meta["dead_files"], [names[5]])
        self.assertEqual(data.meta["failure_file"], names[5])
        self.assertEqual(len(data.timestamps), 5)
        self.assertEqual(data.features.shape, (1, 5, len(FEATURE_ORDER)))
        self.assertTrue(np.all(np.isfinite(data.features)))
        self.assertAlmostEqual(data.rul_hours[0], 50 / 60, places=6)
        self.assertAlmostEqual(data.rul_hours[-1], 10 / 60, places=6)   # BUKAN 0
        self.assertEqual(data.meta["cache_version"], ims.CACHE_VERSION)


class FeatureTests(SimpleTestCase):
    def _t(self):
        return np.arange(ims.IMS_SNAPSHOT_SAMPLES) / ims.IMS_SAMPLE_RATE_HZ

    def test_sine_analytic(self):
        # 1000 Hz x 1.024 s = 1024 siklus bulat -> RMS & bin dominan eksak
        x = 2.0 * np.sin(2 * np.pi * 1000.0 * self._t()) + 0.7
        f = ims.extract_features_ims(x)
        self.assertEqual(list(f.keys()), list(FEATURE_ORDER))
        self.assertAlmostEqual(f["rms"], 2.0 / np.sqrt(2), places=6)
        self.assertAlmostEqual(f["dominant_freq_hz"], 1000.0, delta=1.0)
        self.assertTrue(all(np.isfinite(v) for v in f.values()))

    def test_synthetic_outer_race_peaks_at_bpfo(self):
        ff = compute_fault_frequencies(fr_hz=ims.IMS_RPM / 60.0, **ims.IMS_GEOMETRY)
        t = self._t()
        # carrier resonansi 3 kHz (di dalam band 1-5 kHz) dimodulasi pada BPFO
        x = (1 + 0.8 * np.cos(2 * np.pi * ff.bpfo * t)) * np.cos(2 * np.pi * 3000.0 * t)
        f = ims.extract_features_ims(x)
        self.assertGreater(f["bpfo_amp"], 5 * max(f["bpfi_amp"], f["bsf_amp"]))

    def test_geometry_identity(self):
        ff = compute_fault_frequencies(fr_hz=ims.IMS_RPM / 60.0, **ims.IMS_GEOMETRY)
        self.assertAlmostEqual(ff.bpfo + ff.bpfi, ims.IMS_GEOMETRY["num_balls"] * ims.IMS_RPM / 60.0, places=6)


class CacheRoundTripTests(SimpleTestCase):
    def test_roundtrip(self):
        ts = np.array([0.0, 600.0, 1200.0])
        feats = np.random.default_rng(0).normal(size=(1, 3, len(FEATURE_ORDER)))
        d = ims.TrajectoryData(
            traj_id="IMS-X", timestamps=ts, features=feats, columns=(0,),
            rul_hours=ims.compute_rul_hours(ts),
            hi=ims.compute_health_indicator(feats[0][:, 0], feats[0][:, 1]),
            meta={"traj_id": "IMS-X", "file_list_hash": "abc"},
        )
        with tempfile.TemporaryDirectory() as out:
            p = ims.save_trajectory_cache(d, out)
            back = ims.load_trajectory_cache(p)
        np.testing.assert_array_equal(back.features, d.features)
        np.testing.assert_array_equal(back.rul_hours, d.rul_hours)
        self.assertEqual(back.meta["file_list_hash"], "abc")
        self.assertEqual(back.columns, (0,))


@unittest.skipUnless(os.path.isdir(ims.DEFAULT_IMS_ROOT / "2nd_test" / "2nd_test"),
                     "dataset IMS belum ter-mount di container")
class RealDataTests(SimpleTestCase):
    def test_set2_first_snapshot(self):
        traj = ims.get_trajectory("IMS-S2-B1")
        lst = ims.discover_snapshots(traj, ims.DEFAULT_IMS_ROOT)
        self.assertEqual(len(lst.entries), traj.readme_n_files)      # 984 sesuai Readme
        self.assertEqual(lst.entries[0].name, traj.readme_start)
        self.assertEqual(lst.entries[-1].name, traj.readme_end)
        data = ims.load_snapshot(lst.entries[0].path, traj.n_columns_in_file)
        self.assertEqual(data.shape, (20480, 4))
        f = ims.extract_features_ims(data[:, traj.columns[0]])
        self.assertTrue(all(np.isfinite(v) for v in f.values()))
