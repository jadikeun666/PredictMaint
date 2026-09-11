"""
Helper ekstraksi fitur OFFLINE untuk training script — TIDAK dipakai oleh
pipeline real-time apps.signal_processing (yang menghitung fitur sendiri
saat ingestion, lihat apps.signal_processing.tasks).

Module ini ada supaya training script apps.ml_inference reuse PERSIS
primitive komputasi yang sama dengan pipeline live
(apps.signal_processing.dsp_features, .envelope_features,
.bearing_frequencies) alih-alih implementasi ulang matematikanya
(engineering-rules.md §Code Organization). Hanya menambah glue dari
1 waveform mentah + RPM + geometri -> 9-fitur vector yang dikonsumsi
rf-baseline (ml-pipeline.md §1).
"""
from apps.signal_processing.dsp_features import (
    remove_dc, apply_hann_window, compute_time_domain_features,
    compute_fft, compute_dominant_frequency,
)
from apps.signal_processing.envelope_features import (
    compute_envelope_spectrum, lookup_amplitude_near,
)
from apps.signal_processing.bearing_frequencies import compute_fault_frequencies

# Geometri bearing drive-end CWRU (SKF 6205-2RS JEM), per
# docs/bearing-fault-formulas.md §Known Geometry Values — dicatat di sana
# khusus supaya setiap training/eval script pakai angka yang sama persis.
CWRU_DRIVE_END_GEOMETRY = {
    "num_balls": 9,
    "ball_diameter_mm": 7.94,
    "pitch_diameter_mm": 39.04,
    "contact_angle_deg": 0.0,
}

FEATURE_ORDER = [
    "rms", "kurtosis", "crest_factor", "skewness",
    "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp",
    "dominant_freq_hz",
]


def extract_feature_vector(waveform, sample_rate_hz, rpm):
    """
    Dari 1 window waveform mentah 1.0s + RPM-nya, hitung 9 fitur persis
    yang dikonsumsi rf-baseline (ml-pipeline.md §1), reuse primitive yang
    SAMA dengan pipeline live (signal-processing.md steps 2-8).
    """
    dc_removed = remove_dc(waveform)
    hann_windowed = apply_hann_window(dc_removed)

    time_feats = compute_time_domain_features(dc_removed)
    freqs, magnitude = compute_fft(hann_windowed, sample_rate_hz)
    dominant_freq_hz = compute_dominant_frequency(freqs, magnitude)

    fr_hz = rpm / 60.0
    fault_freqs = compute_fault_frequencies(
        fr_hz=fr_hz,
        num_balls=CWRU_DRIVE_END_GEOMETRY["num_balls"],
        ball_diameter_mm=CWRU_DRIVE_END_GEOMETRY["ball_diameter_mm"],
        pitch_diameter_mm=CWRU_DRIVE_END_GEOMETRY["pitch_diameter_mm"],
        contact_angle_deg=CWRU_DRIVE_END_GEOMETRY["contact_angle_deg"],
    )

    env_freqs, env_spectrum = compute_envelope_spectrum(dc_removed, sample_rate_hz)

    bpfo_amp = lookup_amplitude_near(env_freqs, env_spectrum, fault_freqs.bpfo)
    bpfi_amp = lookup_amplitude_near(env_freqs, env_spectrum, fault_freqs.bpfi)
    bsf_amp = lookup_amplitude_near(env_freqs, env_spectrum, fault_freqs.bsf)
    ftf_amp = lookup_amplitude_near(env_freqs, env_spectrum, fault_freqs.ftf)

    return {
        "rms": time_feats["rms"],
        "kurtosis": time_feats["kurtosis"],
        "crest_factor": time_feats["crest_factor"],
        "skewness": time_feats["skewness"],
        "bpfo_amp": bpfo_amp,
        "bpfi_amp": bpfi_amp,
        "bsf_amp": bsf_amp,
        "ftf_amp": ftf_amp,
        "dominant_freq_hz": dominant_freq_hz,
    }
