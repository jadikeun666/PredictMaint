"""
apps/ml_inference/pu_cache.py

Ekstraksi cache evaluasi PU: untuk tiap file berlabel (HEALTHY/OUTER_RACE/INNER_RACE) ->
4 jendela 1,0 detik -> (a) 9 fitur kanonik (geometri PER KODE dari pu_bearing_labels.json,
RPM terukur dari kanal speed) untuk rf-baseline, (b) gelombang 12 kHz float32 untuk CNN.
Keluaran (out_dir): raw12k.npy (memmap float32) + index.npz (fitur + metadata per jendela).
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np

from apps.ml_inference import pu_dataset as pu

CACHE_VERSION = 1


def _process_file(args):
    path, code, geometry = args
    from apps.ml_inference import features_generic as fg          # lazy: butuh apps.signal_processing
    x, rpm = pu.read_pu_file(path)
    wins = pu.cut_windows(x)[: pu.WINDOWS_PER_FILE]   # 1..4 jendela penuh; sisa < 1 detik dibuang (tidak dipadatkan)
    feats = np.stack([fg.to_vector(fg.extract_features(w, pu.PU_SAMPLE_RATE_HZ, rpm, geometry)) for w in wins])
    raw = np.stack([pu.to_cnn_window(w) for w in wins])
    return feats, raw, rpm


def build_cache(pu_root, out_dir, labels, codes=None, workers=6, limit_files=None, progress=None) -> dict:
    codes = list(codes) if codes else pu.eval_codes(labels)
    jobs, meta = [], []
    for code in codes:
        e = labels[code]
        files = pu.list_files(pu_root, code)
        if limit_files:
            files = files[:limit_files]
        for p in files:
            info = pu.parse_file_name(p.name)
            jobs.append((str(p), code, e["geometry"]))
            meta.append((code, e["label"], e["origin"], info["speed_code"], info["torque_code"], info["force_code"], info["index"]))
    n_files, W = len(jobs), pu.WINDOWS_PER_FILE          # W = batas atas jendela per file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_tmp, final_tmp = out_dir / "raw12k.tmp.npy", out_dir / "raw12k.final.tmp.npy"
    raw = np.lib.format.open_memmap(raw_tmp, mode="w+", dtype=np.float32, shape=(n_files * W, pu.CNN_WINDOW_SAMPLES))
    feats_all = np.zeros((n_files * W, 9), dtype=np.float64)
    rpm_all = np.zeros(n_files * W, dtype=np.float64)
    counts, pos, t0 = [], 0, time.time()
    try:
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("fork")) as ex:
            for i, (f, r, rpm) in enumerate(ex.map(_process_file, jobs, chunksize=4)):
                k = len(r)
                raw[pos:pos + k] = r
                feats_all[pos:pos + k] = f
                rpm_all[pos:pos + k] = rpm
                counts.append(k)
                pos += k
                if progress and ((i + 1) % 100 == 0 or i + 1 == n_files):
                    progress(i + 1, n_files, time.time() - t0)
        n = pos
        raw.flush()
        final = np.lib.format.open_memmap(final_tmp, mode="w+", dtype=np.float32, shape=(n, pu.CNN_WINDOW_SAMPLES))
        for s0 in range(0, n, 1000):
            e0 = min(s0 + 1000, n)
            final[s0:e0] = raw[s0:e0]
        final.flush()
        del final, raw
        os.remove(raw_tmp)
    except BaseException:
        for p in (raw_tmp, final_tmp):
            if p.exists():
                p.unlink()
        raise

    counts = np.array(counts)
    rep = lambda k: np.repeat(np.array([m[k] for m in meta]), counts)
    index = {
        "features": feats_all[:n], "rpm": rpm_all[:n],
        "code": rep(0).astype("U5"), "label": rep(1).astype("U12"), "origin": rep(2).astype("U12"),
        "speed_code": rep(3).astype("U2"), "torque_code": rep(4).astype("U2"), "force_code": rep(5).astype("U2"),
        "file_index": rep(6).astype(np.int64),
        "window_index": np.concatenate([np.arange(c) for c in counts]).astype(np.int64),
        "cache_version": np.array(CACHE_VERSION),
    }
    sane = np.array([pu.rpm_sanity(r, s) for r, s in zip(index["rpm"], index["speed_code"])])
    index["rpm_sane"] = sane
    tmp_idx = out_dir / "index.tmp.npz"
    np.savez_compressed(tmp_idx, **index)
    os.replace(final_tmp, out_dir / "raw12k.npy")
    os.replace(tmp_idx, out_dir / "index.npz")
    hist = {int(k): int((counts == k).sum()) for k in sorted(set(counts.tolist()))}
    return {"n_files": n_files, "n_windows": int(n), "windows_per_file": hist, "seconds": time.time() - t0,
            "n_rpm_insane": int((~sane).sum()), "out_dir": str(out_dir)}
