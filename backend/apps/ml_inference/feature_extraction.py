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

CWRU_DRIVE_END_GEOMETRY = {
    "num_balls": 9,
    "ball_diameter_mm": 7.94,
    "pitch_diameter_mm": 39.04,
    "contact_angle_deg": 0.0,
}

# 9 fitur KANONIK persis sesuai kontrak `features` dict yang dikirim
# apps.signal_processing.tasks ke run_inference (claude.md — FIXED, tidak
# berubah). Ini yang benar-benar dikirim lewat MQTT->Celery di live path.
FEATURE_ORDER = [
    "rms", "kurtosis", "crest_factor", "skewness",
    "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp",
    "dominant_freq_hz",
]

# Fitur TURUNAN — dihitung HANYA dari 9 nilai kanonik di atas, TIDAK
# butuh data tambahan apa pun dari signal_processing. Ditambahkan setelah
# rf-baseline v1.0 (2026-09-11) menunjukkan confusion BALL<->OUTER_RACE:
# bpfo_amp milik BALL ikut terangkat (energi broadband akibat ball-spin
# randomization — fenomena dikenal di literatur bearing-fault), sehingga
# magnitude absolut kurang diskriminatif. Rasio relatif ("frekuensi mana
# yang dominan dibanding total energi fault") lebih tahan terhadap
# variasi skala akibat severity/load dibanding magnitude mentah.
ENGINEERED_FEATURE_ORDER = [
    "bpfo_ratio", "bpfi_ratio", "bsf_ratio", "ftf_ratio",
    "total_fault_energy", "dominant_fault_ratio",
]

# Urutan LENGKAP yang benar-benar dikonsumsi model (9 kanonik + 6
# turunan). run_inference WAJIB membangun vector dengan urutan persis
# ini (baca dari model_registry.json entry-nya, field "feature_order"),
# bukan hardcode ulang — supaya training script dan inference selalu
# sinkron kalau MODEL_FEATURE_ORDER berubah lagi di masa depan.
MODEL_FEATURE_ORDER = FEATURE_ORDER + ENGINEERED_FEATURE_ORDER


def extract_feature_vector(waveform, sample_rate_hz, rpm):
    """
    Dari 1 window waveform mentah 1.0s + RPM-nya, hitung 9 FITUR KANONIK
    (FEATURE_ORDER) persis yang dikonsumsi rf-baseline (ml-pipeline.md
    §1), reuse primitive yang SAMA dengan pipeline live
    (signal-processing.md steps 2-8). Tidak termasuk fitur turunan —
    itu tanggung jawab engineer_features(), dipanggil terpisah supaya
    run_inference bisa reuse fungsi ini dengan tepat 9 nilai yang sama
    yang diterimanya dari signal_processing (bukan menghitung ulang
    waveform, run_inference tidak punya akses waveform mentah).
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


def engineer_features(base: dict) -> dict:
    """
    Fitur turunan (ENGINEERED_FEATURE_ORDER) dari 9 fitur kanonik —
    dihitung HANYA dari nilai yang sudah ada di `base` (dict berisi
    minimal 4 key bpfo_amp/bpfi_amp/bsf_amp/ftf_amp), tanpa butuh
    waveform atau data tambahan apa pun. Ini yang membuat fungsi ini
    aman dipanggil ulang oleh run_inference (yang cuma punya `features`
    dict dari signal_processing, bukan waveform mentah).

    Dipanggil dari kode yang SUDAH memvalidasi tidak ada None di antara
    bpfo/bpfi/bsf/ftf_amp (lihat keputusan missing-feature-at-inference
    — window dengan fitur null di-skip SEBELUM sampai ke sini).
    """
    eps = 1e-9
    bpfo = base["bpfo_amp"]
    bpfi = base["bpfi_amp"]
    bsf = base["bsf_amp"]
    ftf = base["ftf_amp"]
    total = bpfo + bpfi + bsf + ftf

    return {
        "bpfo_ratio": bpfo / (total + eps),
        "bpfi_ratio": bpfi / (total + eps),
        "bsf_ratio": bsf / (total + eps),
        "ftf_ratio": ftf / (total + eps),
        "total_fault_energy": total,
        "dominant_fault_ratio": max(bpfo, bpfi, bsf, ftf) / (total + eps),
    }
