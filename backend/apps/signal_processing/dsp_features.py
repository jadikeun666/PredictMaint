"""
Pure-numpy/scipy DSP feature-extraction helpers for the signal_processing
pipeline (signal-processing.md §Pipeline steps 2-5, 8).

Kept separate from tasks.py (which owns DB/Celery orchestration) so these
are independently unit-testable with synthetic signals, and so a future
offline training/eval script could reuse the exact same feature
computation without importing Celery/Django (engineering-rules.md
§Code Organization — same rationale as bearing_frequencies.py).

No float32 truncation anywhere — everything stays float64 (signal-
processing.md §Numerical Precision).
"""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann


def remove_dc(waveform: np.ndarray) -> np.ndarray:
    """Step 2: DC removal — subtract the mean of the window."""
    waveform = waveform.astype(np.float64, copy=False)
    return waveform - np.mean(waveform)


def apply_hann_window(dc_removed: np.ndarray) -> np.ndarray:
    """Step 3: Hann window applied before FFT to reduce spectral leakage
    (references.md — Steven W. Smith, ch.9). Returns a NEW array; the
    caller's DC-removed raw signal is left untouched, since time-domain
    stats (step 4) must use the raw DC-removed signal, not this one."""
    n = dc_removed.shape[0]
    window = hann(n, sym=False)
    return dc_removed * window


def compute_time_domain_features(dc_removed_raw: np.ndarray) -> dict:
    """
    Step 4: time-domain statistical features, computed on the DC-removed
    RAW signal (NOT the Hann-windowed copy) — signal-processing.md is
    explicit that windowing distorts amplitude-based statistics.

    rms          = sqrt(mean(x^2))
    kurtosis     = fourth standardized moment, Fisher/EXCESS convention
                   (i.e. normal distribution -> kurtosis == 0)
    crest_factor = max(|x|) / rms
    skewness     = third standardized moment
    """
    x = dc_removed_raw.astype(np.float64, copy=False)
    n = x.shape[0]

    rms = np.sqrt(np.mean(x ** 2))

    std = np.std(x)  # population std (ddof=0) — standard for these moments
    if std == 0:
        # degenerate flat-line window (e.g. all-zero waveform); avoid
        # divide-by-zero, report 0.0 rather than NaN/inf — a flat signal
        # has no meaningful higher-order shape.
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = float(np.mean(((x - np.mean(x)) / std) ** 3))
        # Fisher (excess) kurtosis: subtract 3 so Gaussian == 0
        kurtosis = float(np.mean(((x - np.mean(x)) / std) ** 4) - 3.0)

    max_abs = np.max(np.abs(x))
    crest_factor = float(max_abs / rms) if rms != 0 else 0.0

    return {
        "rms": float(rms),
        "kurtosis": kurtosis,
        "crest_factor": crest_factor,
        "skewness": skewness,
    }


def compute_fft(hann_windowed: np.ndarray, sample_rate_hz: float):
    """
    Step 5: FFT of the Hann-windowed signal via scipy.fft.rfft.
    NOT zero-padded to a power of two (signal-processing.md §Numerical
    Precision — would distort frequency-bin alignment against
    BPFO/BPFI/BSF/FTF lookups later).

    Returns (freqs_hz, magnitude) as float64 arrays, magnitude being
    |rfft(x)| (not power, not normalized — dominant-frequency bin lookup
    and fault-frequency amplitude lookup both just need relative
    magnitude, consistent with how signal-processing.md describes both
    steps 6-8 as reading "magnitude").
    """
    n = hann_windowed.shape[0]
    spectrum = rfft(hann_windowed)
    freqs = rfftfreq(n, d=1.0 / sample_rate_hz)
    magnitude = np.abs(spectrum).astype(np.float64)
    return freqs.astype(np.float64), magnitude


def compute_dominant_frequency(freqs_hz: np.ndarray, magnitude: np.ndarray) -> float:
    """
    Step 8: frequency bin with the highest magnitude in the raw FFT,
    EXCLUDING DC (index 0, freq 0 Hz) — signal-processing.md is explicit
    about excluding DC here (DC bin is near-zero after step 2 anyway, but
    excluded per spec regardless of magnitude, not just as a side-effect
    of DC removal).
    """
    if freqs_hz.shape[0] <= 1:
        return 0.0
    idx_excl_dc = np.argmax(magnitude[1:]) + 1
    return float(freqs_hz[idx_excl_dc])
