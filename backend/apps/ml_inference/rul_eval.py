"""
apps/ml_inference/rul_eval.py

Bagian NUMERIK MURNI (tanpa torch, tanpa Django) untuk training/evaluasi
lstm-rul (ml-pipeline.md §2): pembentukan jendela 6 titik, normalisasi fitur,
metrik (MAE, RMSE, skor asimetris gaya PHM08), baseline sederhana, dan
pembagian fold leave-one-trajectory-out. Dipisah dari model torch supaya
bisa diuji tanpa torch dan tanpa mengimpor apa pun yang berat.

KEPUTUSAN DESAIN (dilaporkan di addendum Giliran 4):
- Fitur: 9 kanonik kanal PRIMER -> signed-log1p -> z-score. Statistik z-score
  dihitung dari lintasan LATIH saja per fold (tanpa kebocoran).
- Skor PHM08 asimetris diadaptasi ke jam: error d dinyatakan dalam % umur
  lintasan uji (d = 100*(pred-asli)/panjang_lintasan_jam), lalu
  s = exp(-d/13)-1 bila d<0 (terlalu dini), s = exp(d/10)-1 bila d>=0
  (terlalu OPTIMIS, RUL prediksi > asli; dihukum lebih berat karena
  penyebut 10 < 13). Skor = rata-rata per titik (bukan jumlah) supaya
  tidak bergantung pada panjang lintasan. Konstanta 13/10 dari PHM08.
- Baseline HI-ekstrapolasi memakai informasi yang SAMA dengan LSTM (jendela
  6 titik terakhir) dan HI dinormalisasi dengan min/max lintasan LATIH
  (kausal; min-max lintasan penuh akan membocorkan snapshot terakhir).
"""
from dataclasses import dataclass

import numpy as np

SEQ_LEN = 6                  # ml-pipeline.md §2 Sequence Length
GAP_MINUTES = 15.0           # jendela yang melintasi celah > ini ditandai (tidak dibuang)
PHM_A_EARLY = 13.0
PHM_A_LATE = 10.0
HI_FAILURE_LEVEL = 1.0
HI_WEIGHTS = (0.5, 0.5)      # sama dengan ims_dataset.HI_WEIGHT_*


# --- normalisasi fitur ------------------------------------------------------
def signed_log1p(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray


def fit_scaler(feature_arrays) -> Scaler:
    """feature_arrays: list array (N_i, F) MENTAH dari lintasan latih."""
    X = np.concatenate([signed_log1p(a) for a in feature_arrays], axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)      # pengaman fitur konstan
    return Scaler(mean, std)


def apply_scaler(sc: Scaler, feats) -> np.ndarray:
    return (signed_log1p(feats) - sc.mean) / sc.std


# --- jendela ----------------------------------------------------------------
@dataclass
class WindowSet:
    X: np.ndarray            # (M, seq_len, F)
    y: np.ndarray            # (M,) RUL jam pada titik terakhir jendela
    last_idx: np.ndarray     # (M,) indeks titik terakhir di lintasan
    crosses_gap: np.ndarray  # (M,) bool


def make_windows(feats, timestamps_s, rul_hours, seq_len=SEQ_LEN) -> WindowSet:
    feats = np.asarray(feats, dtype=np.float64)
    ts = np.asarray(timestamps_s, dtype=np.float64)
    rul = np.asarray(rul_hours, dtype=np.float64)
    n = len(feats)
    if n < seq_len:
        raise ValueError(f"lintasan hanya {n} titik < seq_len {seq_len}")
    idx = np.arange(seq_len - 1, n)
    X = np.stack([feats[i - seq_len + 1:i + 1] for i in idx])
    gaps = np.diff(ts) / 60.0 > GAP_MINUTES              # gaps[k] = celah antara titik k dan k+1
    crosses = np.array([bool(gaps[i - seq_len + 1:i].any()) for i in idx])
    return WindowSet(X, rul[idx], idx, crosses)


# --- metrik -----------------------------------------------------------------
def regression_metrics(pred, true, traj_len_hours) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true
    d = 100.0 * err / float(traj_len_hours)
    s = np.where(d < 0, np.exp(-np.minimum(d, 0) / PHM_A_EARLY) - 1.0,
                 np.exp(np.maximum(d, 0) / PHM_A_LATE) - 1.0)
    last = true <= 0.10 * traj_len_hours
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "phm": float(np.mean(s)),
        "mae_last10pct": float(np.mean(np.abs(err[last]))) if last.any() else float("nan"),
        "bias": float(np.mean(err)),             # > 0 = rata-rata terlalu optimis
        "n": int(len(true)),
    }


# --- baseline ---------------------------------------------------------------
def baseline_constant(train_rul_all, n_test) -> np.ndarray:
    return np.full(n_test, float(np.mean(np.concatenate(train_rul_all))))


@dataclass
class HIParams:
    rms_min: float
    rms_max: float
    kurt_min: float
    kurt_max: float
    fallback_hours: float
    cap_hours: float


def fit_hi_params(train_raw_feats, train_rul_all, i_rms=0, i_kurt=1) -> HIParams:
    X = np.concatenate(train_raw_feats, axis=0)
    r = np.concatenate(train_rul_all)
    return HIParams(float(X[:, i_rms].min()), float(X[:, i_rms].max()),
                    float(X[:, i_kurt].min()), float(X[:, i_kurt].max()),
                    float(r.mean()), float(r.max()))


def _mm(v, lo, hi):
    return (v - lo) / (hi - lo) if hi - lo > 0 else np.zeros_like(v)


def hi_extrapolation_predict(raw_feats, timestamps_s, last_idx, p: HIParams,
                             i_rms=0, i_kurt=1, seq_len=SEQ_LEN):
    """
    Per jendela: HI titik (min-max dari lintasan latih) -> regresi linear HI vs
    waktu pada 6 titik -> RUL = (1 - HI_sekarang) / kemiringan. Bila
    kemiringan <= 0 (tidak ada tren naik) -> fallback (rata-rata RUL latih).
    Hasil di-clip ke [0, RUL maksimum latih]. Kembalikan (pred, pakai_fallback).
    """
    raw = np.asarray(raw_feats, dtype=np.float64)
    ts = np.asarray(timestamps_s, dtype=np.float64)
    w_r, w_k = HI_WEIGHTS
    hi = w_r * _mm(raw[:, i_rms], p.rms_min, p.rms_max) + w_k * _mm(raw[:, i_kurt], p.kurt_min, p.kurt_max)
    pred = np.empty(len(last_idx))
    fb = np.zeros(len(last_idx), dtype=bool)
    for j, i in enumerate(last_idx):
        t = (ts[i - seq_len + 1:i + 1] - ts[i]) / 3600.0       # jam relatif (<= 0)
        slope, intercept = np.polyfit(t, hi[i - seq_len + 1:i + 1], 1)
        if slope > 1e-9:
            pred[j] = max(0.0, (HI_FAILURE_LEVEL - intercept) / slope)
        else:
            pred[j] = p.fallback_hours
            fb[j] = True
    return np.clip(pred, 0.0, p.cap_hours), fb


# --- pembagian fold ---------------------------------------------------------
SCHEMES = ("loto", "leave_set_out", "loto_excl_set3")


def build_folds(trajs, scheme):
    """
    trajs: list (traj_id, set_no). Kembalikan list (fold_name, test_ids, train_ids).
    - loto           : leave-one-trajectory-out (spec ml-pipeline.md §2.1), 4 fold
    - leave_set_out  : tahan SEMUA lintasan satu set sekaligus (lintasan satu
                       eksperimen/poros tidak muncul di latih dan uji bersamaan)
    - loto_excl_set3 : LOTO tanpa Set 3 sama sekali (uji sensitivitas, Set 3
                       menyimpang dari rentang Readme)
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme tidak dikenal: {scheme}")
    pool = [t for t in trajs if not (scheme == "loto_excl_set3" and t[1] == 3)]
    if scheme == "leave_set_out":
        groups = {}
        for tid, s in pool:
            groups.setdefault(f"set{s}", []).append(tid)
    else:
        groups = {tid: [tid] for tid, _ in pool}
    folds = []
    for name, test_ids in groups.items():
        train_ids = [tid for tid, _ in pool if tid not in test_ids]
        assert not set(test_ids) & set(train_ids)
        if not train_ids:
            raise ValueError("fold tanpa lintasan latih")
        folds.append((name, list(test_ids), train_ids))
    return folds
