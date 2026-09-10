"""
Shared bearing characteristic-fault-frequency formulas.

Locked source: docs/bearing-fault-formulas.md — do not duplicate these
formulas anywhere else (engineering-rules.md §Code Organization). This
module has NO Django/ORM dependency on purpose, so it can be imported
both by `apps.signal_processing` (Celery worker) and by any offline
training/eval script without needing a full Django app context.

Symbols (bearing-fault-formulas.md §Symbols):
    fr   shaft rotation frequency, Hz  (= RPM / 60)
    n    number of rolling elements (balls)     -> bearings.num_balls
    d    rolling element diameter, mm           -> bearings.ball_diameter_mm
    D    pitch diameter, mm                     -> bearings.pitch_diameter_mm
    phi  contact angle, degrees (0 for deep-groove ball bearings)
                                                 -> bearings.contact_angle_deg
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FaultFrequencies:
    bpfo: float
    bpfi: float
    bsf: float
    ftf: float


def compute_fault_frequencies(fr_hz, num_balls, ball_diameter_mm,
                               pitch_diameter_mm, contact_angle_deg) -> FaultFrequencies:
    """
    Compute BPFO / BPFI / BSF / FTF (Hz) per bearing-fault-formulas.md.

    All geometry args must be non-null, non-zero-pitch-diameter floats —
    callers (the signal_processing pipeline) are responsible for checking
    for missing bearing geometry BEFORE calling this (see
    signal-processing.md §Pipeline step 7 / prd.md A2: missing geometry
    means these fields are skipped entirely, not computed with a
    fallback/default value here). This function does not fabricate
    defaults for missing geometry — it will raise if given None/zero.
    """
    if None in (fr_hz, num_balls, ball_diameter_mm, pitch_diameter_mm, contact_angle_deg):
        raise ValueError(
            "compute_fault_frequencies() requires all geometry values to be non-null; "
            "the caller must check bearing geometry completeness before calling this "
            "(signal-processing.md step 7 / prd.md A2 — never fabricate a default)."
        )

    n = float(num_balls)
    d = float(ball_diameter_mm)
    D = float(pitch_diameter_mm)
    phi_rad = math.radians(float(contact_angle_deg))
    fr = float(fr_hz)

    ratio = d / D
    cos_phi = math.cos(phi_rad)

    bpfo = (n / 2.0) * fr * (1.0 - ratio * cos_phi)
    bpfi = (n / 2.0) * fr * (1.0 + ratio * cos_phi)
    bsf = (D / (2.0 * d)) * fr * (1.0 - (ratio ** 2) * (cos_phi ** 2))
    ftf = (fr / 2.0) * (1.0 - ratio * cos_phi)

    return FaultFrequencies(bpfo=bpfo, bpfi=bpfi, bsf=bsf, ftf=ftf)


def bpfo_plus_bpfi_identity_holds(fr_hz, num_balls, freqs: FaultFrequencies,
                                   rel_tol=1e-9) -> bool:
    """
    Sanity identity from bearing-fault-formulas.md: for a stationary outer
    race, BPFO + BPFI == n * fr. Used by the unit test below; exposed here
    (not re-implemented in the test file) so the pipeline could also reuse
    it later for a geometry sanity-check without duplicating the formula
    (engineering-rules.md §Code Organization).
    """
    expected = float(num_balls) * float(fr_hz)
    actual = freqs.bpfo + freqs.bpfi
    return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=1e-9)
