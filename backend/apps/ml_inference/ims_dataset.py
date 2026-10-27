"""
apps/ml_inference/ims_dataset.py

Pemuat dataset IMS run-to-failure + pembangun fitur per snapshot + label RUL
dan Health Indicator (HI) untuk `lstm-rul` (ml-pipeline.md §2). OFFLINE saja —
tidak dipakai jalur inference live.

Reuse penuh (TANPA mengubah) primitive apps.signal_processing.{dsp_features,
envelope_features, bearing_frequencies}, pola sama dengan train_rf_baseline /
feature_extraction.py. Yang BERBEDA dari extract_feature_vector() CWRU:
geometri, RPM, dan sample rate IMS (extract_feature_vector meng-hardcode
geometri CWRU, dan file itu tidak boleh diubah).

SUMBER FAKTA (engineering-rules.md "no fabricated values") — semua konstanta
IMS di bawah dibuktikan dari "Readme Document for IMS Bearing Data.pdf":
  - sampling rate 20 kHz, 20.480 titik/snapshot (1-second snapshot)
  - RPM konstan 2000
  - Set 1: 8 kanal, Bearing 3 = Ch 5&6, Bearing 4 = Ch 7&8; gagal: inner
    race (B3), roller element (B4); 2.156 file; 22 Okt 2003 12:06:24 s/d
    25 Nov 2003 23:39:56
  - Set 2: 4 kanal (1 per bearing); gagal: outer race (B1); 984 file;
    12 Feb 2004 10:32:39 s/d 19 Feb 2004 06:22:39
  - Set 3: 4 kanal; gagal: outer race (B3); Readme: 4.448 file; 4 Mar 2004
    09:27:46 s/d 4 Apr 2004 19:01:57
Geometri (d, D, n, phi) dari bearing-fault-formulas.md (Rexnord ZA-2115).

TEMUAN DATA vs README (dilaporkan, bukan ditebak):
1. Folder Set 3 di disk berisi 6.324 file sampai 18 Apr 2004, lebih banyak dari
   Readme (4.448 file s/d 4 Apr 2004 19:01:57). Di dalam batas Readme TIDAK ada
   degradasi terukur (RMS datar 0,066-0,069); degradasi baru muncul di file
   sesudahnya. Keputusan: lintasan Set 3 memakai SEMUA file di disk
   (`use_all_files=True`); waktu kegagalan = file terakhir di disk. Ini
   menyimpang dari rentang tanggal Readme (tetapi konsisten dengan pernyataan
   Readme bahwa kerusakan terjadi di AKHIR eksperimen) -> dicatat sbg deviasi.
2. Snapshot "sinyal mati" (std ~0,001 di semua kanal lintasan; ditemukan di 2
   file terakhir Set 2 dan file terakhir Set 3) adalah akuisisi yang berhenti,
   bukan degradasi. Snapshot itu dibuang dari INPUT, tetapi waktu kegagalan
   tetap file terakhir di disk (RUL snapshot valid terakhir > 0).
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from apps.ml_inference.feature_extraction import FEATURE_ORDER
from apps.signal_processing.bearing_frequencies import compute_fault_frequencies
from apps.signal_processing.dsp_features import (
    apply_hann_window, compute_dominant_frequency, compute_fft,
    compute_time_domain_features, remove_dc,
)
from apps.signal_processing.envelope_features import (
    compute_envelope_spectrum, lookup_amplitude_near,
)

# --- Konstanta IMS (sumber: Readme IMS + bearing-fault-formulas.md) ---------
IMS_SAMPLE_RATE_HZ = 20000
IMS_SNAPSHOT_SAMPLES = 20480
IMS_RPM = 2000.0
IMS_GEOMETRY = {
    "num_balls": 16,
    "ball_diameter_mm": 8.41,
    "pitch_diameter_mm": 71.50,
    "contact_angle_deg": 15.17,
}

# Bobot HI: ml-pipeline.md §2 hanya menyebut "normalized_weighted_sum" tanpa
# angka bobot -> keputusan implementasi: bobot sama rata (dilaporkan di
# addendum Giliran 4).
# Ambang sinyal mati: std < 0,01 di SEMUA kanal lintasan. Std snapshot normal
# yang teramati >= ~0,05; snapshot mati teramati ~0,001.
DEAD_SIGNAL_STD_THRESHOLD = 0.01
CACHE_VERSION = 2

HI_WEIGHT_RMS = 0.5
HI_WEIGHT_KURTOSIS = 0.5

DEFAULT_IMS_ROOT = Path(os.environ.get("IMS_DATA_ROOT", "/dataset/ims"))
# Sama kedalaman dengan model_registry.py -> backend/ml_artifacts/...
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "ml_artifacts" / "ims_features"

_SNAPSHOT_NAME_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


@dataclass(frozen=True)
class Trajectory:
    traj_id: str
    set_no: int
    folder: str                # relatif terhadap ims_root
    bearing_no: int
    failure_mode: str          # teks Readme
    columns: tuple             # indeks kolom 0-based; elemen pertama = kanal PRIMER
    n_columns_in_file: int
    readme_n_files: int
    readme_start: str          # nama file format timestamp (Readme)
    readme_end: str            # nama file format timestamp (Readme) = failure
    use_all_files: bool = False  # True: abaikan batas readme_end, pakai semua file di disk


# 4 lintasan run-to-failure yang gagalnya terdokumentasi di Readme.
# Bearing lain di tiap set TIDAK gagal -> tersensor, tidak punya label RUL.
# Readme tidak menyebut kanal mana x/y pada Set 1 -> kanal pertama tiap
# pasangan dipakai sebagai primer, kanal kedua tetap diekstrak.
TRAJECTORIES = (
    Trajectory("IMS-S1-B3", 1, "1st_test/1st_test", 3, "inner race defect",
               (4, 5), 8, 2156, "2003.10.22.12.06.24", "2003.11.25.23.39.56"),
    Trajectory("IMS-S1-B4", 1, "1st_test/1st_test", 4, "roller element defect",
               (6, 7), 8, 2156, "2003.10.22.12.06.24", "2003.11.25.23.39.56"),
    Trajectory("IMS-S2-B1", 2, "2nd_test/2nd_test", 1, "outer race failure",
               (0,), 4, 984, "2004.02.12.10.32.39", "2004.02.19.06.22.39"),
    Trajectory("IMS-S3-B3", 3, "3rd_test/4th_test/txt", 3, "outer race failure",
               (2,), 4, 4448, "2004.03.04.09.27.46", "2004.04.04.19.01.57", True),
)


def get_trajectory(traj_id: str) -> Trajectory:
    for t in TRAJECTORIES:
        if t.traj_id == traj_id:
            return t
    raise KeyError(f"traj_id tidak dikenal: {traj_id} (tersedia: {[t.traj_id for t in TRAJECTORIES]})")


# --- Timestamp & listing ----------------------------------------------------
def parse_snapshot_timestamp(name: str) -> float:
    """'YYYY.MM.DD.HH.MM.SS' -> epoch detik (zona dianggap UTC; hanya selisih yang bermakna)."""
    if not _SNAPSHOT_NAME_RE.match(name):
        raise ValueError(f"Nama file bukan format timestamp snapshot IMS: {name!r}")
    y, mo, d, h, mi, s = (int(p) for p in name.split("."))
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp()


class Snapshot(NamedTuple):
    name: str
    path: Path
    ts: float
    size: int


@dataclass
class SnapshotListing:
    entries: list              # list[Snapshot], terurut naik by ts, hanya ts <= readme_end
    n_in_folder: int           # semua file di folder
    n_ignored_non_snapshot: int
    n_after_readme_end: int    # jumlah file di luar batas readme_end (dikecualikan ATAU dipakai)
    after_end_used: bool = False


def discover_snapshots(traj: Trajectory, ims_root) -> SnapshotListing:
    folder = Path(ims_root) / traj.folder
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder lintasan {traj.traj_id} tidak ada: {folder} "
            f"(cek mount dataset IMS / --ims-root)"
        )
    end_ts = parse_snapshot_timestamp(traj.readme_end)
    kept, n_total, n_ignored, n_after = [], 0, 0, 0
    with os.scandir(folder) as it:
        for e in it:
            if not e.is_file():
                continue
            n_total += 1
            if not _SNAPSHOT_NAME_RE.match(e.name):
                n_ignored += 1
                continue
            ts = parse_snapshot_timestamp(e.name)
            if ts > end_ts:
                n_after += 1
                if not traj.use_all_files:
                    continue
            kept.append(Snapshot(e.name, Path(e.path), ts, e.stat().st_size))
    kept.sort(key=lambda s: s.ts)
    return SnapshotListing(kept, n_total, n_ignored, n_after, traj.use_all_files)


def file_list_hash(entries) -> str:
    """Hash daftar file persis (nama + ukuran) — untuk training_dataset_version_hash."""
    h = hashlib.sha256()
    for s in entries:
        h.update(f"{s.name}:{s.size}\n".encode())
    return h.hexdigest()


def load_snapshot(path, n_columns: int, expected_rows: int = IMS_SNAPSHOT_SAMPLES) -> np.ndarray:
    """Baca 1 file ASCII snapshot -> array float64 (rows, n_columns). Gagal keras kalau bentuk salah."""
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.shape != (expected_rows, n_columns):
        raise ValueError(
            f"Bentuk snapshot {path} = {arr.shape}, diharapkan ({expected_rows}, {n_columns})"
        )
    return arr


# --- Fitur per snapshot -----------------------------------------------------
def extract_features_ims(waveform) -> dict:
    """
    9 fitur KANONIK (FEATURE_ORDER) dari 1 snapshot 1 kanal IMS. Langkah sama
    dengan signal-processing.md (DC removal, Hann, time-domain pada sinyal
    DC-removed, FFT, envelope, lookup BPFO/BPFI/BSF/FTF), tetapi dengan
    geometri/RPM/sample-rate IMS.
    """
    x = np.asarray(waveform, dtype=np.float64)
    dc_removed = remove_dc(x)
    hann_windowed = apply_hann_window(dc_removed)

    tf = compute_time_domain_features(dc_removed)
    freqs, magnitude = compute_fft(hann_windowed, IMS_SAMPLE_RATE_HZ)
    dominant_freq_hz = compute_dominant_frequency(freqs, magnitude)

    ff = compute_fault_frequencies(fr_hz=IMS_RPM / 60.0, **IMS_GEOMETRY)
    env_freqs, env_spectrum = compute_envelope_spectrum(dc_removed, IMS_SAMPLE_RATE_HZ)

    out = {
        "rms": tf["rms"],
        "kurtosis": tf["kurtosis"],
        "crest_factor": tf["crest_factor"],
        "skewness": tf["skewness"],
        "bpfo_amp": lookup_amplitude_near(env_freqs, env_spectrum, ff.bpfo),
        "bpfi_amp": lookup_amplitude_near(env_freqs, env_spectrum, ff.bpfi),
        "bsf_amp": lookup_amplitude_near(env_freqs, env_spectrum, ff.bsf),
        "ftf_amp": lookup_amplitude_near(env_freqs, env_spectrum, ff.ftf),
        "dominant_freq_hz": dominant_freq_hz,
    }
    return {k: float(v) for k, v in out.items()}


def features_to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float64)


def is_dead_snapshot(data, columns) -> bool:
    """True kalau SEMUA kanal lintasan nyaris datar (akuisisi berhenti)."""
    return bool(np.all(np.std(data[:, list(columns)], axis=0) < DEAD_SIGNAL_STD_THRESHOLD))


def _extract_one(args):
    """Worker proses: 1 file -> (array (n_kanal, 9), dead). Top-level supaya bisa di-pickle."""
    path, n_columns, columns = args
    data = load_snapshot(path, n_columns)
    if is_dead_snapshot(data, columns):
        return np.full((len(columns), len(FEATURE_ORDER)), np.nan), True
    feats = np.stack([features_to_vector(extract_features_ims(data[:, c])) for c in columns])
    return feats, False


# --- Label RUL & HI ---------------------------------------------------------
def compute_rul_hours(timestamps, failure_ts=None) -> np.ndarray:
    """
    RUL (jam) = waktu kegagalan - waktu snapshot. Dari timeline, bukan estimasi.
    failure_ts default = timestamp snapshot terakhir; boleh lebih besar kalau
    snapshot terakhir di disk dibuang (sinyal mati) tapi waktu kegagalan tetap.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size == 0:
        raise ValueError("timestamps kosong")
    if np.any(np.diff(ts) < 0):
        raise ValueError("timestamps harus terurut naik")
    ref = ts[-1] if failure_ts is None else float(failure_ts)
    if ref < ts[-1]:
        raise ValueError("failure_ts lebih awal dari snapshot terakhir")
    return (ref - ts) / 3600.0


def _minmax(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo <= 0.0:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def compute_health_indicator(rms, kurtosis) -> np.ndarray:
    """HI = minmax( w1*minmax(rms) + w2*minmax(kurtosis) ) (ml-pipeline.md §2)."""
    s = HI_WEIGHT_RMS * _minmax(rms) + HI_WEIGHT_KURTOSIS * _minmax(kurtosis)
    return _minmax(s)


# --- Lintasan lengkap + cache ----------------------------------------------
@dataclass
class TrajectoryData:
    traj_id: str
    timestamps: np.ndarray     # (N,) epoch detik
    features: np.ndarray       # (n_kanal, N, 9) float64; kanal 0 = primer
    columns: tuple
    rul_hours: np.ndarray      # (N,)
    hi: np.ndarray             # (N,) dari kanal primer
    meta: dict

    @property
    def primary(self) -> np.ndarray:
        return self.features[0]


def build_trajectory(traj: Trajectory, listing: SnapshotListing, ims_root=None,
                     workers: int = 8, limit=None, progress=None) -> TrajectoryData:
    """Ekstrak fitur seluruh snapshot lintasan (paralel). `limit` hanya untuk uji waktu."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    entries = listing.entries[:limit] if limit else listing.entries
    if not entries:
        raise ValueError(f"{traj.traj_id}: tidak ada snapshot")
    jobs = [(str(s.path), traj.n_columns_in_file, traj.columns) for s in entries]

    results = []
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("fork")) as ex:
        for i, r in enumerate(ex.map(_extract_one, jobs, chunksize=8), start=1):
            results.append(r)
            if progress and (i % 200 == 0 or i == len(jobs)):
                progress(i, len(jobs))

    feats_all = np.stack([r[0] for r in results], axis=1)          # (n_kanal, N, 9)
    dead = np.array([r[1] for r in results], dtype=bool)
    if dead.all():
        raise ValueError(f"{traj.traj_id}: semua snapshot sinyal mati")
    valid = ~dead
    ts_all = np.array([s.ts for s in entries], dtype=np.float64)
    failure_ts = float(ts_all[-1])                                 # file terakhir di disk (termasuk yang mati)
    ts = ts_all[valid]
    feats = feats_all[:, valid, :]
    i_rms, i_kurt = FEATURE_ORDER.index("rms"), FEATURE_ORDER.index("kurtosis")
    meta = {
        "cache_version": CACHE_VERSION,
        "traj_id": traj.traj_id,
        "file_list_hash": file_list_hash(entries),
        "n_files": len(entries),
        "n_valid": int(valid.sum()),
        "n_dead_excluded": int(dead.sum()),
        "dead_files": [entries[i].name for i in np.where(dead)[0]],
        "failure_file": entries[-1].name,
        "failure_ts": failure_ts,
        "readme_n_files": traj.readme_n_files,
        "n_after_readme_end": listing.n_after_readme_end,
        "after_readme_end_used": bool(listing.after_end_used),
        "n_in_folder": listing.n_in_folder,
        "first_file": entries[0].name,
        "last_valid_file": entries[np.where(valid)[0][-1]].name,
        "columns": list(traj.columns),
        "sample_rate_hz": IMS_SAMPLE_RATE_HZ,
        "rpm": IMS_RPM,
        "feature_order": list(FEATURE_ORDER),
        "hi_weights": [HI_WEIGHT_RMS, HI_WEIGHT_KURTOSIS],
        "truncated_by_limit": bool(limit),
    }
    return TrajectoryData(
        traj_id=traj.traj_id, timestamps=ts, features=feats, columns=traj.columns,
        rul_hours=compute_rul_hours(ts, failure_ts=failure_ts),
        hi=compute_health_indicator(feats[0][:, i_rms], feats[0][:, i_kurt]),
        meta=meta,
    )


def cache_path(out_dir, traj_id: str) -> Path:
    return Path(out_dir) / f"{traj_id}.npz"


def save_trajectory_cache(data: TrajectoryData, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = cache_path(out_dir, data.traj_id)
    tmp = out_dir / f"{data.traj_id}.tmp.npz"
    np.savez_compressed(
        tmp, timestamps=data.timestamps, features=data.features,
        rul_hours=data.rul_hours, hi=data.hi,
        columns=np.array(data.columns, dtype=np.int64),
        meta_json=np.array(json.dumps(data.meta, sort_keys=True)),
    )
    os.replace(tmp, final)
    return final


def load_trajectory_cache(path) -> TrajectoryData:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        return TrajectoryData(
            traj_id=meta["traj_id"], timestamps=z["timestamps"], features=z["features"],
            columns=tuple(int(c) for c in z["columns"]), rul_hours=z["rul_hours"],
            hi=z["hi"], meta=meta,
        )
