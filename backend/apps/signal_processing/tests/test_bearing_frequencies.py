"""
Unit test for the BPFO + BPFI = n * fr sanity identity
(engineering-rules.md §Testing, bearing-fault-formulas.md §Characteristic
Defect Frequencies), using CWRU's documented drive-end bearing geometry
(bearing-fault-formulas.md §Known Geometry Values: SKF 6205-2RS JEM).
"""
import math
import unittest

from apps.signal_processing.bearing_frequencies import (
    compute_fault_frequencies,
    bpfo_plus_bpfi_identity_holds,
)

# CWRU SKF 6205-2RS JEM (drive end) — bearing-fault-formulas.md Known Geometry Values
CWRU_DRIVE_END = dict(num_balls=9, ball_diameter_mm=7.94, pitch_diameter_mm=39.04,
                       contact_angle_deg=0)
# CWRU ~1797 RPM condition (mqtt-architecture.md §Simulator example)
FR_HZ = 1797.0 / 60.0


class BpfoPlusBpfiIdentityTest(unittest.TestCase):
    def test_identity_holds_for_cwru_drive_end_geometry(self):
        freqs = compute_fault_frequencies(fr_hz=FR_HZ, **CWRU_DRIVE_END)
        self.assertTrue(bpfo_plus_bpfi_identity_holds(FR_HZ, CWRU_DRIVE_END["num_balls"], freqs))
        self.assertAlmostEqual(freqs.bpfo + freqs.bpfi,
                                CWRU_DRIVE_END["num_balls"] * FR_HZ, places=9)

    def test_identity_fails_if_num_balls_is_wrong(self):
        freqs = compute_fault_frequencies(fr_hz=FR_HZ, **CWRU_DRIVE_END)
        wrong_n = CWRU_DRIVE_END["num_balls"] + 1
        self.assertFalse(
            math.isclose(freqs.bpfo + freqs.bpfi, wrong_n * FR_HZ, rel_tol=1e-9)
        )

    def test_missing_geometry_raises_not_fabricates_default(self):
        with self.assertRaises(ValueError):
            compute_fault_frequencies(fr_hz=FR_HZ, num_balls=None,
                                       ball_diameter_mm=7.94, pitch_diameter_mm=39.04,
                                       contact_angle_deg=0)

    def test_ftf_less_than_fr_for_deep_groove_bearing(self):
        freqs = compute_fault_frequencies(fr_hz=FR_HZ, **CWRU_DRIVE_END)
        self.assertLess(freqs.ftf, FR_HZ)


if __name__ == "__main__":
    unittest.main()
