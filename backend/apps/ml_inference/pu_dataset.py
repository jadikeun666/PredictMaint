"""
apps/ml_inference/pu_dataset.py

Pemuat dataset Paderborn (PU) untuk evaluasi lintas-dataset klasifikasi bearing
(ml-pipeline.md §1.4). SEMUA fakta di bawah dibuktikan dari dokumentasi PU
(measuring_log_K001.pdf, PDF per-kode) dan dari data, bukan dari ingatan:
  - getaran `vibration_1`, 64 kHz nominal (measuring log); panjang file 256.000-258.669
    sampel pada ~4,0 detik (jitter laju efektif kecil; laju nominal dipakai)
  - kanal `speed` (4 kHz) = RPM terukur; kode kondisi pada nama file:
    N09 = 900 rpm, N15 = 1500 rpm (terbukti: rata-rata terukur 899,7 dan 1499,6)
  - label/asal/geometri per kode: data/pu_bearing_labels.json (turunan PDF, lihat pu_labels.py)
Satuan amplitudo PU TIDAK tercatat di file (kolom Unit kosong); std getaran sehat ~0,37
vs CWRU ~0,065 (~5,6x) — dicatat, tidak diasumsikan.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

PU_SAMPLE_RATE_HZ = 64000
CNN_SAMPLE_RATE_HZ = 12000                       # ml-pipeline.md §Common Sample Rate
WINDOW_SAMPLES = PU_SAMPLE_RATE_HZ               # jendela 1,0 detik
CNN_WINDOW_SAMPLES = CNN_SAMPLE_RATE_HZ
WINDOWS_PER_FILE = 4                             # ~4,0 detik per file, tanpa overlap
EVAL_LABELS = ("HEALTHY", "OUTER_RACE", "INNER_RACE")   # PU tidak punya BALL; MIXED dikeluarkan
NOMINAL_RPM = {"09": 900.0, "15": 1500.0}        # kode kondisi pada nama file (+ measuring log); hanya untuk SANITY
RPM_TOLERANCE = 0.03

LABELS_PATH = Path(__file__).resolve().parent / "data" / "pu_bearing_labels.json"
DEFAULT_PU_ROOT = Path(os.environ.get("PU_DATA_ROOT", "/dataset/pu"))
_FILE_RE = re.compile(r"^N(\d{2})_M(\d{2})_F(\d{2})_(K[A-Z0-9]{3})_(\d+)\.mat$")


def load_labels(path=LABELS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)["codes"]


def eval_codes(labels: dict):
    return sorted(c for c, e in labels.items() if e["label"] in EVAL_LABELS)


def parse_file_name(name: str) -> dict:
    m = _FILE_RE.match(name)
    if not m:
        raise ValueError(f"nama file PU tidak dikenal: {name!r}")
    return {"speed_code": m.group(1), "torque_code": m.group(2), "force_code": m.group(3),
            "code": m.group(4), "index": int(m.group(5))}


def list_files(pu_root, code):
    d = Path(pu_root) / code
    if not d.is_dir():
        raise FileNotFoundError(f"folder kode PU tidak ada: {d}")
    files = [p for p in d.glob(f"N??_M??_F??_{code}_*.mat")]
    def key(p):
        i = parse_file_name(p.name)
        return (i["speed_code"], i["torque_code"], i["force_code"], i["index"])
    return sorted(files, key=key)


def read_pu_file(path):
    """Return (vibration float64, rpm_terukur float) dari satu file .mat PU."""
    m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    keys = [k for k in m if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{path}: kunci teratas tidak tunggal {keys}")
    top = m[keys[0]]
    # loadmat(squeeze_me=True) mengubah array struct 1 elemen menjadi satu objek -> atleast_1d
    chans = {y.Name: np.asarray(y.Data, dtype=np.float64) for y in np.atleast_1d(top.Y) if isinstance(y.Name, str)}
    if "vibration_1" not in chans or "speed" not in chans:
        raise ValueError(f"{path}: kanal vibration_1/speed tidak ada (ada: {sorted(chans)})")
    return chans["vibration_1"], float(np.mean(chans["speed"]))


def cut_windows(x, window_samples=WINDOW_SAMPLES) -> np.ndarray:
    x = np.asarray(x)
    n = len(x) // window_samples
    if n < 1:
        raise ValueError(f"sinyal {len(x)} sampel < satu jendela {window_samples}")
    return x[: n * window_samples].reshape(n, window_samples)


def to_cnn_window(window, fs_hz=PU_SAMPLE_RATE_HZ) -> np.ndarray:
    """Resample satu jendela 1,0 detik ke 12 kHz (scipy.signal.resample, sesuai spec). float32."""
    n_out = int(round(len(window) * CNN_SAMPLE_RATE_HZ / fs_hz))
    return resample(np.asarray(window, dtype=np.float64), n_out).astype(np.float32)


def rpm_sanity(rpm: float, speed_code: str, tol=RPM_TOLERANCE) -> bool:
    nominal = NOMINAL_RPM[speed_code]
    return abs(rpm - nominal) / nominal <= tol
