import numpy as np
from django.test import SimpleTestCase

from apps.ml_inference import rul_relative as rr


class RelativeTests(SimpleTestCase):
    def test_zero_at_own_baseline(self):
        raw = np.tile(np.array([2.0, 50.0, 7.0]), (150, 1))
        np.testing.assert_allclose(rr.relative_features(raw, 100), 0.0, atol=1e-12)

    def test_level_offset_between_bearings_is_removed(self):
        t = np.linspace(0, 1, 200)
        a = np.column_stack([100 + 400 * t ** 3, 80 + 200 * t ** 3])
        b = a * 3.0                                           # bearing lain: tingkat dasar 3x lebih tinggi
        np.testing.assert_allclose(rr.relative_features(a, 100), rr.relative_features(b, 100), atol=0.02)

    def test_degradation_shows_as_positive_shift(self):
        raw = np.column_stack([np.concatenate([np.full(100, 10.0), np.linspace(10, 100, 100)])])
        rel = rr.relative_features(raw, 100)
        self.assertAlmostEqual(float(rel[:100].max()), 0.0, places=9)
        self.assertGreater(float(rel[-1, 0]), 1.5)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            rr.relative_features(np.ones((50, 3)), 100)

    def test_zscore_and_mask(self):
        rng = np.random.default_rng(0)
        a = [rng.normal(3, 2, size=(200, 4)), np.column_stack([rng.normal(3, 2, size=(200, 3)), np.full(200, 1.0)])]
        sc = rr.fit_zscore(a)
        z = rr.apply_zscore(sc, a[0])
        self.assertTrue(np.all(np.isfinite(z)))
        np.testing.assert_array_equal(rr.window_mask([5, 99, 100, 101], 100), [False, False, True, True])
