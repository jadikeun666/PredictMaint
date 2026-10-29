"""
apps/ml_inference/rul_alert_policy.py

Simulasi OFFLINE kebijakan alert berbasis RUL (Sesi C, Giliran 1) — numpy murni,
tanpa torch/Django, supaya mudah diuji. Dipakai command `analyze_rul_alert_policy`
untuk memilih kebijakan dari prediksi out-of-fold hasil leave-one-trajectory-out.

Kebijakan = penghalusan median kausal (K jendela) -> level severity per jendela
(ambang inklusif, sama dengan alerting.severity.rul_severity) -> opsional
konfirmasi tren (RUL terhalus harus menurun dibanding `lag` jendela lalu) ->
persistensi (level harus bertahan M jendela berturut-turut).

Hasil alert pertama untuk level S (CRITICAL/HIGH) diklasifikasikan:
  HIT        : RUL ASLI saat alert pertama berada di [lead_min_S, factor * ambang_S]
  PREMATURE  : RUL asli > factor * ambang_S      (alarm terlalu dini = false alarm)
  LATE       : RUL asli < lead_min_S             (tidak cukup waktu menjadwalkan)
  MISS       : tidak ada alert sampai kegagalan
  NA         : umur lintasan < lead_min_S (tidak mungkin HIT; dikeluarkan dari tingkat itu)
lead_min dari scheduling.md §3 (target jadwal URGENT 48 jam, HIGH 7 hari).

ATURAN SELEKSI (ditetapkan SEBELUM melihat hasil): layak bila tingkat PREMATURE
<= max_premature di CRITICAL dan HIGH, dan rata-rata tingkat HIT kedua level
>= min_hit; dari yang layak pilih HIT tertinggi, lalu paling sederhana. Bila
tidak ada yang layak -> None (rekomendasi: batasi jalur RUL ke WARNING).
"""
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

LEVEL_NAMES = {3: "CRITICAL", 2: "HIGH", 1: "WARNING", 0: "INFO"}
EVAL_LEVELS = (3, 2)                                   # level yang memicu Work Order
DEFAULT_LEAD_MIN_HOURS = {3: 48.0, 2: 168.0}
DEFAULT_PREMATURE_FACTOR = 2.0
GRID_K = (1, 6, 12, 24)
GRID_M = (1, 6, 12)


@dataclass(frozen=True)
class Thresholds:
    critical: float
    high: float
    warning: float

    def hours_for(self, rank: int) -> float:
        return {3: self.critical, 2: self.high, 1: self.warning}[rank]


@dataclass(frozen=True)
class Policy:
    k_smooth: int = 1
    m_persist: int = 1
    trend: bool = False

    @property
    def name(self) -> str:
        return f"K{self.k_smooth}-M{self.m_persist}-{'trend' if self.trend else 'notrend'}"


BASELINE_POLICY = Policy(1, 1, False)                  # P0 = perilaku saat ini (opsi A)


@dataclass
class Case:
    traj_id: str
    seed: int
    true_rul: np.ndarray       # (n,) RUL asli per jendela, urut waktu
    pred: np.ndarray           # (n,) prediksi RUL (jam) per jendela
    traj_hours: float


def smooth_causal(pred, k: int) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    if k <= 1:
        return pred.copy()
    out = np.empty_like(pred)
    for i in range(min(k - 1, len(pred))):                # tepi awal: jendela yang masih pendek
        out[i] = np.median(pred[:i + 1])
    if len(pred) >= k:
        out[k - 1:] = np.median(sliding_window_view(pred, k), axis=1)
    return out


def severity_rank(x, th: Thresholds) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    r = np.zeros(len(x), dtype=int)
    r[x <= th.warning] = 1
    r[x <= th.high] = 2
    r[x <= th.critical] = 3
    return r


def confirmed_levels(pred, policy: Policy, th: Thresholds) -> np.ndarray:
    """Level severity terkonfirmasi per jendela (0..3)."""
    sm = smooth_causal(pred, policy.k_smooth)
    lvl = severity_rank(sm, th)
    if policy.trend:
        lag = max(policy.k_smooth, 6)
        ok = np.zeros(len(sm), dtype=bool)
        if len(sm) > lag:
            ok[lag:] = sm[lag:] < sm[:-lag]
        lvl = np.where(ok, lvl, 0)
    m = policy.m_persist
    if m <= 1:
        return lvl
    conf = np.zeros_like(lvl)
    if len(lvl) >= m:
        conf[m - 1:] = sliding_window_view(lvl, m).min(axis=1)
    return conf


def first_alert_index(conf, rank: int):
    idx = np.flatnonzero(np.asarray(conf) >= rank)
    return int(idx[0]) if len(idx) else None


def classify_outcome(true_rul, first, h, lead_min, factor, traj_hours) -> str:
    if traj_hours < lead_min:
        return "NA"
    if first is None:
        return "MISS"
    tr = float(true_rul[first])
    if tr > factor * h:
        return "PREMATURE"
    if tr < lead_min:
        return "LATE"
    return "HIT"


def evaluate_policy(cases, policy: Policy, th: Thresholds, factor=DEFAULT_PREMATURE_FACTOR,
                    lead_min=None) -> dict:
    lead_min = lead_min or DEFAULT_LEAD_MIN_HOURS
    out = {"policy": policy.name, "levels": {}, "per_case": []}
    counts = {r: {"HIT": 0, "PREMATURE": 0, "LATE": 0, "MISS": 0, "NA": 0} for r in EVAL_LEVELS}
    for c in cases:
        conf = confirmed_levels(c.pred, policy, th)
        row = {"traj_id": c.traj_id, "seed": c.seed}
        for r in EVAL_LEVELS:
            first = first_alert_index(conf, r)
            o = classify_outcome(c.true_rul, first, th.hours_for(r), lead_min[r], factor, c.traj_hours)
            counts[r][o] += 1
            row[LEVEL_NAMES[r]] = {"outcome": o, "first_idx": first,
                                   "true_rul_at_alert": None if first is None else float(c.true_rul[first])}
        out["per_case"].append(row)
    for r in EVAL_LEVELS:
        n_eval = sum(v for k, v in counts[r].items() if k != "NA")
        d = dict(counts[r])
        d["n_eval"] = n_eval
        for k in ("HIT", "PREMATURE", "LATE", "MISS"):
            d[k.lower() + "_rate"] = (counts[r][k] / n_eval) if n_eval else None
        out["levels"][LEVEL_NAMES[r]] = d
    return out


def build_grid():
    return [Policy(k, m, t) for k in GRID_K for m in GRID_M for t in (False, True)]


def select_policy(results, max_premature=0.25, min_hit=0.5):
    """results: list of evaluate_policy() output. Return (nama_terpilih | None, alasan)."""
    def rates(res):
        c, h = res["levels"]["CRITICAL"], res["levels"]["HIGH"]
        return c, h

    eligible = []
    for res in results:
        c, h = rates(res)
        if c["n_eval"] == 0 or h["n_eval"] == 0:
            continue
        if c["premature_rate"] > max_premature or h["premature_rate"] > max_premature:
            continue
        hit = (c["hit_rate"] + h["hit_rate"]) / 2.0
        if hit < min_hit:
            continue
        p = _policy_from_name(res["policy"])
        eligible.append((-hit, p.k_smooth + p.m_persist + (1 if p.trend else 0), res["policy"]))
    if not eligible:
        return None, (f"tidak ada kebijakan dengan PREMATURE <= {max_premature:.0%} di CRITICAL dan HIGH "
                      f"sekaligus rata-rata HIT >= {min_hit:.0%}")
    eligible.sort()
    return eligible[0][2], f"HIT rata-rata {-eligible[0][0]:.0%}, kriteria PREMATURE <= {max_premature:.0%} terpenuhi"


def _policy_from_name(name: str) -> Policy:
    k, m, t = name.split("-")
    return Policy(int(k[1:]), int(m[1:]), t == "trend")
