"""
apps/ml_inference/rul_relative.py

Fitur RELATIF terhadap baseline bearing itu sendiri (numpy murni) — kandidat
`lstm-rul` v1.1. Motivasi: model v1.0 mempelajari tingkat absolut fitur, padahal
tingkat dasar bearing sehat berbeda antar bearing (mis. RMS sehat ~0,11 vs ~0,07),
sehingga bearing sehat dinilai hampir gagal (Sesi C, Giliran 1).

Transformasi: f = signed_log1p(fitur mentah); f_rel = f - median(f[:B]) dengan B
snapshot PERTAMA bearing itu (= rasio perubahan terhadap kondisi awalnya sendiri).
Prediksi baru boleh keluar setelah baseline lengkap (titik terakhir jendela >= B).
Baseline di sistem live = B titik pertama bearing sejak `installed_at`.
"""
import numpy as np

from apps.ml_inference.rul_eval import Scaler, signed_log1p

BASELINE_POINTS = 100


def relative_features(raw, baseline_points: int = BASELINE_POINTS) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if len(raw) < baseline_points:
        raise ValueError(f"butuh >= {baseline_points} titik untuk baseline, hanya {len(raw)}")
    f = signed_log1p(raw)
    return f - np.median(f[:baseline_points], axis=0)


def fit_zscore(rel_arrays) -> Scaler:
    X = np.concatenate(rel_arrays, axis=0)
    std = X.std(axis=0)
    return Scaler(X.mean(axis=0), np.where(std < 1e-12, 1.0, std))


def apply_zscore(sc: Scaler, rel) -> np.ndarray:
    return (np.asarray(rel, dtype=np.float64) - sc.mean) / sc.std


def window_mask(last_idx, baseline_points: int = BASELINE_POINTS) -> np.ndarray:
    return np.asarray(last_idx) >= baseline_points
