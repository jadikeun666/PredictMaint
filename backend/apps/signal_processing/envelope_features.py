"""
Envelope spectrum + fault-frequency amplitude lookup
(signal-processing.md §Pipeline steps 6-7).

Pure numpy/scipy, no Django/ORM/DB dependency — same rationale as
dsp_features.py and bearing_frequencies.py (engineering-rules.md
§Code Organization): independently testable, reusable by offline
scripts later.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import rfft, rfftfreq

DEFAULT_ENVELOPE_BANDPASS_LOW_HZ = 1000.0
DEFAULT_ENVELOPE_BANDPASS_HIGH_HZ = 5000.0
ENVELOPE_BANDPASS_ORDER = 4
FAULT_FREQ_TOLERANCE_FRACTION = 0.02


def compute_envelope_spectrum(dc_removed_raw: np.ndarray, sample_rate_hz: float,
                               band_low_hz: float = DEFAULT_ENVELOPE_BANDPASS_LOW_HZ,
                               band_high_hz: float = DEFAULT_ENVELOPE_BANDPASS_HIGH_HZ):
    nyquist = sample_rate_hz / 2.0
    high = min(band_high_hz, nyquist * 0.99)
    low = min(band_low_hz, high * 0.5)

    sos_low = low / nyquist
    sos_high = high / nyquist

    b, a = butter(ENVELOPE_BANDPASS_ORDER, [sos_low, sos_high], btype="bandpass")
    filtered = filtfilt(b, a, dc_removed_raw)

    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)

    n = envelope.shape[0]
    env_spectrum = np.abs(rfft(envelope)).astype(np.float64)
    env_freqs = rfftfreq(n, d=1.0 / sample_rate_hz).astype(np.float64)

    return env_freqs, env_spectrum


def lookup_amplitude_near(env_freqs_hz: np.ndarray, env_spectrum: np.ndarray,
                           target_freq_hz: float,
                           tolerance_fraction: float = FAULT_FREQ_TOLERANCE_FRACTION) -> float:
    if target_freq_hz <= 0:
        return 0.0
    half_width = target_freq_hz * tolerance_fraction
    mask = (env_freqs_hz >= target_freq_hz - half_width) & (env_freqs_hz <= target_freq_hz + half_width)
    if not np.any(mask):
        return 0.0
    return float(np.max(env_spectrum[mask]))
