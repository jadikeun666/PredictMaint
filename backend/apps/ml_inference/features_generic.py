"""
apps/ml_inference/features_generic.py

9 fitur kanonik (FEATURE_ORDER) dari satu jendela, dengan sample rate, RPM, dan geometri bearing
sebagai PARAMETER. Memakai ulang primitive apps.signal_processing tanpa mengubahnya (langkah sama
dengan signal-processing.md). Dipakai untuk dataset selain CWRU/IMS (mis. PU); fungsi CWRU
(feature_extraction) dan IMS (ims_dataset) tetap tidak diubah — ada test paritas terhadap IMS.
"""
import numpy as np

from apps.ml_inference.feature_extraction import FEATURE_ORDER
from apps.signal_processing.bearing_frequencies import compute_fault_frequencies
from apps.signal_processing.dsp_features import (
    apply_hann_window, compute_dominant_frequency, compute_fft,
    compute_time_domain_features, remove_dc,
)
from apps.signal_processing.envelope_features import compute_envelope_spectrum, lookup_amplitude_near


def extract_features(waveform, fs_hz, rpm, geometry) -> dict:
    """geometry: num_balls, ball_diameter_mm, pitch_diameter_mm, contact_angle_deg."""
    x = np.asarray(waveform, dtype=np.float64)
    dc = remove_dc(x)
    hann = apply_hann_window(dc)
    tf = compute_time_domain_features(dc)
    freqs, magnitude = compute_fft(hann, fs_hz)
    dominant = compute_dominant_frequency(freqs, magnitude)
    ff = compute_fault_frequencies(
        fr_hz=rpm / 60.0, num_balls=geometry["num_balls"], ball_diameter_mm=geometry["ball_diameter_mm"],
        pitch_diameter_mm=geometry["pitch_diameter_mm"], contact_angle_deg=geometry["contact_angle_deg"])
    env_f, env_s = compute_envelope_spectrum(dc, fs_hz)
    out = {
        "rms": tf["rms"], "kurtosis": tf["kurtosis"], "crest_factor": tf["crest_factor"], "skewness": tf["skewness"],
        "bpfo_amp": lookup_amplitude_near(env_f, env_s, ff.bpfo),
        "bpfi_amp": lookup_amplitude_near(env_f, env_s, ff.bpfi),
        "bsf_amp": lookup_amplitude_near(env_f, env_s, ff.bsf),
        "ftf_amp": lookup_amplitude_near(env_f, env_s, ff.ftf),
        "dominant_freq_hz": dominant,
    }
    return {k: float(out[k]) for k in FEATURE_ORDER}


def to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float64)
