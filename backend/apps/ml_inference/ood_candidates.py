"""
apps/ml_inference/ood_candidates.py

Sesi E, Giliran 1: kandidat penjaga aplikabilitas model (OOD) + protokol evaluasi
kalibrasi-CWRU-saja. Modul MURNI (numpy + scikit-learn; tanpa Django/Celery/DB) supaya
bisa diuji tanpa konteks Django dan, bila dipilih, dipakai ulang apa adanya oleh jalur
inference (Giliran 2).

KONFIGURASI TERKUNCI: CONFIG di bawah di-hash (config_hash). Skrip evaluasi
(`eval_ood_guard`) menolak berjalan bila hash berbeda dari yang tercatat saat protokol
disetujui. Jangan diubah setelah melihat hasil PU/IMS.

Ruang fitur S9: signed-log1p dari 9 fitur kanonik, lalu z-score memakai statistik DATA FIT
(bukan data uji). Skor selalu "makin besar = makin di luar domain".
"""
import hashlib
import json

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

# Sama dengan apps.ml_inference.feature_extraction.FEATURE_ORDER (command memeriksa kesamaannya).
FEATURES = (
    "rms", "kurtosis", "crest_factor", "skewness",
    "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp",
    "dominant_freq_hz",
)
CANDIDATES = ("G1", "G2", "G3", "G4")
STAT_CANDIDATES = ("G2", "G3", "G4")   # urutan = urutan kesederhanaan (tie-break)
SD_FLOOR = 1e-6

CONFIG = {
    "name": "ood-guard-eval-v1",
    "features": list(FEATURES),
    "transform": "signed_log1p_then_zscore_with_fit_set_stats",
    "sd_floor": SD_FLOOR,
    "candidates": {
        "G1": "reject jika ada fitur di luar [min,max] data fit (mentah); tau=0, tanpa kuantil",
        "G2": {"score": "min_class_mahalanobis", "cov": "ledoit_wolf_pooled_within_class"},
        "G3": {"score": "mean_knn_distance", "k": 5},
        "G4": {"score": "neg_score_samples", "n_estimators": 200, "seed": 42,
               "contamination": "auto"},
    },
    "quantiles": [0.95, 0.99],
    "quantile_method": "linear",
    "threshold_source": "pooled_leave_one_group_out_scores",
    "groups": "fault: instance (B007..OR021 = 9); HEALTHY: per file Time_Normal_L0..L3 (4, bocor sebagian)",
    "nested_frr": "tau_i dari skor grup j!=i yang dinilai guard yang di-fit tanpa {i,j}; FRR_i dari guard tanpa i",
    "frr_cap_macro": 0.10,
    "selection_q": 0.95,
    "selection_metric": "rata-rata 3 shift dari deteksi makro antar-grup pada selection_q",
    "tie_tol": 0.02,
    "tie_order": list(STAT_CANDIDATES),
    "shifts": {
        "amp2": {"mul": 2.0, "cols": ["rms", "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp"]},
        "amp5": {"mul": 5.0, "cols": ["rms", "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp"]},
        "impulsive": {"mul_crest": 2.5, "add_kurtosis": 10.0},
    },
    "rf_alarm_min_conf": 0.6,
    "rf_critical_min_conf": 0.85,
    "rf_alarm_classes": ["BALL", "INNER_RACE", "OUTER_RACE"],
    "ims_phases": {"early_elapsed_lt": 0.5, "late_elapsed_ge": 0.9},
    "pu_filter": "rpm_sane",
}


def config_hash(config=None):
    cfg = CONFIG if config is None else config
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def s9(X):
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


class Guard:
    """Satu kandidat. fit(X9, y) lalu score(X9) -> skor OOD (makin besar makin asing)."""

    def __init__(self, kind):
        if kind not in CANDIDATES:
            raise ValueError("kandidat tidak dikenal: %s" % kind)
        self.kind = kind

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if self.kind == "G1":
            self.lo = X.min(axis=0)
            self.hi = X.max(axis=0)
            return self
        Z = s9(X)
        self.mu = Z.mean(axis=0)
        self.sd = np.maximum(Z.std(axis=0), SD_FLOOR)
        Zs = (Z - self.mu) / self.sd
        if self.kind == "G2":
            y = np.asarray(y)
            classes = sorted(set(y.tolist()))
            cents, resid = [], []
            for c in classes:
                m = Zs[y == c]
                cent = m.mean(axis=0)
                cents.append(cent)
                resid.append(m - cent)
            lw = LedoitWolf(assume_centered=True).fit(np.vstack(resid))
            self.prec = lw.precision_
            self.cent = np.vstack(cents)
        elif self.kind == "G3":
            self.k = CONFIG["candidates"]["G3"]["k"]
            self.nn = NearestNeighbors(n_neighbors=self.k).fit(Zs)
        else:
            c = CONFIG["candidates"]["G4"]
            self.iso = IsolationForest(n_estimators=c["n_estimators"], random_state=c["seed"],
                                       contamination="auto").fit(Zs)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.kind == "G1":
            span = np.maximum(self.hi - self.lo, 1e-12)
            exceed = np.maximum(np.maximum(self.lo - X, X - self.hi), 0.0) / span
            return exceed.max(axis=1)
        Zs = (s9(X) - self.mu) / self.sd
        if self.kind == "G2":
            diff = Zs[:, None, :] - self.cent[None, :, :]
            m2 = np.einsum("nck,kl,ncl->nc", diff, self.prec, diff)
            return np.sqrt(np.maximum(m2, 0.0)).min(axis=1)
        if self.kind == "G3":
            d, _ = self.nn.kneighbors(Zs)
            return d.mean(axis=1)
        return -self.iso.score_samples(Zs)


def tau_for(kind, pool, q):
    if kind == "G1":
        return 0.0
    return float(np.quantile(np.asarray(pool, dtype=np.float64), q))


def apply_shift(X, spec):
    """spec: salah satu entri CONFIG['shifts']. Mengembalikan salinan."""
    X = np.array(X, dtype=np.float64, copy=True)
    idx = {f: i for i, f in enumerate(FEATURES)}
    if "mul" in spec:
        for f in spec["cols"]:
            X[:, idx[f]] *= spec["mul"]
    else:
        X[:, idx["crest_factor"]] *= spec["mul_crest"]
        X[:, idx["kurtosis"]] += spec["add_kurtosis"]
    return X


def evaluate_candidate(X, y, groups, kind, quantiles=None, shifts=None):
    """
    Evaluasi leave-one-group-out + nested-tau untuk satu kandidat (hanya data CWRU).
    X: (n,9) mentah; y: kelas; groups: id grup (>= 3 grup berbeda).
    Mengembalikan dict:
      group_ids, loo_scores {g: array}, final_guard (fit semua), final_tau {q: float},
      per q: frr {g: float}, frr_macro, frr_max, detection {shift: {g: float}},
      detection_macro {shift: float}, detection_mean.
    """
    quantiles = CONFIG["quantiles"] if quantiles is None else quantiles
    shifts = CONFIG["shifts"] if shifts is None else shifts
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    gids = sorted(set(groups.tolist()))
    if len(gids) < 3:
        raise ValueError("butuh >= 3 grup")
    mask = {g: groups == g for g in gids}

    guards = {}   # guard di-fit tanpa g
    loo = {}
    for g in gids:
        keep = ~mask[g]
        guards[g] = Guard(kind).fit(X[keep], y[keep])
        loo[g] = guards[g].score(X[mask[g]])

    # pool_i: skor grup j!=i oleh guard tanpa {i,j}; satu fit melayani pasangan (i,j) sekaligus
    pools = {g: [] for g in gids}
    for a in range(len(gids)):
        for b in range(a + 1, len(gids)):
            i, j = gids[a], gids[b]
            keep = ~(mask[i] | mask[j])
            gd = Guard(kind).fit(X[keep], y[keep])
            pools[j].append(gd.score(X[mask[i]]))
            pools[i].append(gd.score(X[mask[j]]))

    pooled_loo = np.concatenate([loo[g] for g in gids])
    final_guard = Guard(kind).fit(X, y)
    out = {"kind": kind, "group_ids": gids, "loo_scores": loo, "final_guard": final_guard,
           "final_tau": {}, "per_q": {}}
    shifted = {name: {g: apply_shift(X[mask[g]], spec) for g in gids} for name, spec in shifts.items()}
    shifted_scores = {name: {g: guards[g].score(shifted[name][g]) for g in gids} for name in shifts}

    for q in quantiles:
        out["final_tau"][q] = tau_for(kind, pooled_loo, q)
        frr, det = {}, {name: {} for name in shifts}
        for g in gids:
            tau_g = tau_for(kind, np.concatenate(pools[g]), q)
            frr[g] = float(np.mean(loo[g] > tau_g))
            for name in shifts:
                det[name][g] = float(np.mean(shifted_scores[name][g] > tau_g))
        det_macro = {name: float(np.mean(list(det[name].values()))) for name in shifts}
        out["per_q"][q] = {
            "frr": frr,
            "frr_macro": float(np.mean(list(frr.values()))),
            "frr_max": float(max(frr.values())),
            "detection": det,
            "detection_macro": det_macro,
            "detection_mean": float(np.mean(list(det_macro.values()))),
        }
    return out


def select_candidate(results):
    """
    Aturan terkunci (CONFIG): di antara G2/G3/G4 dengan FRR makro <= frr_cap pada selection_q,
    pilih deteksi_mean tertinggi; bila ada yang berselisih <= tie_tol dari yang tertinggi,
    pilih yang terdahulu di tie_order. results: {kind: evaluate_candidate(...)}.
    Mengembalikan (kind atau None, alasan).
    """
    q = CONFIG["selection_q"]
    cap = CONFIG["frr_cap_macro"]
    tol = CONFIG["tie_tol"]
    eligible = [k for k in CONFIG["tie_order"]
                if k in results and results[k]["per_q"][q]["frr_macro"] <= cap]
    if not eligible:
        return None, "tidak ada kandidat statistik dengan FRR makro <= %.2f pada q=%.2f" % (cap, q)
    best = max(results[k]["per_q"][q]["detection_mean"] for k in eligible)
    for k in eligible:
        if results[k]["per_q"][q]["detection_mean"] >= best - tol:
            return k, "eligible=%s; deteksi terbaik %.4f; dipilih yang paling sederhana dalam toleransi %.2f" % (
                eligible, best, tol)
    return None, "tak terjangkau"
