"""Sesi D G3b: DIAGNOSTIK pergeseran domain CWRU -> PU. BACA-SAJA.

PERINGATAN INTEGRITAS: bagian B memilih/membandingkan himpunan fitur sambil
melihat hasil PU. Itu hanya diagnostik. Model hasil bagian B TIDAK boleh
diregistrasi atau dilaporkan sebagai bukti generalisasi zero-shot
(cross_dataset_pu_accuracy harus berasal dari model yang dikunci SEBELUM
melihat PU). Tidak ada registry/model/cache yang diubah.
"""
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from django.core.management.base import BaseCommand, CommandError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from apps.ml_inference import feature_extraction as fe
from apps.ml_inference.management.commands import train_rf_baseline as trb

CACHE = Path("/app/ml_artifacts/pu_cache/index.npz")
OUT_DIR = Path("/app/ml_artifacts/pu_eval")
LAB3 = ["HEALTHY", "OUTER_RACE", "INNER_RACE"]
ALL = list(fe.MODEL_FEATURE_ORDER)
BASE = list(fe.FEATURE_ORDER)

VARIANTS = {
    "S0 semua15": ALL,
    "S1 bebas-skala": ["kurtosis", "crest_factor", "skewness", "bpfo_ratio", "bpfi_ratio",
                       "bsf_ratio", "ftf_ratio", "dominant_fault_ratio"],
    "S2 S1-tanpa-ftf": ["kurtosis", "crest_factor", "skewness", "bpfo_ratio", "bpfi_ratio",
                        "bsf_ratio", "dominant_fault_ratio"],
    "S3 rasio-fault": ["bpfo_ratio", "bpfi_ratio", "bsf_ratio", "dominant_fault_ratio"],
}


def mk():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=42, n_jobs=1)


def pstat(a):
    return "%12.4g  [%.4g - %.4g]" % (np.percentile(a, 50), np.percentile(a, 5), np.percentile(a, 95))


def load_cwru():
    rows, labels, groups = [], [], []
    for fname, (mat_id, cls, load) in sorted(trb.FILE_MANIFEST.items()):
        de_key = "X%s_DE_time" % mat_id
        mat = sio.loadmat(trb.DATASET_ROOT / fname, variable_names=[de_key, "X%sRPM" % mat_id])
        wf = mat[de_key].squeeze().astype(np.float64)
        rpm = trb._load_rpm(mat, mat_id, load, fname)
        w_len = trb.WINDOW_SAMPLES
        for w in range(len(wf) // w_len):
            base = fe.extract_feature_vector(wf[w * w_len:(w + 1) * w_len], trb.SAMPLE_RATE_HZ, rpm)
            full = {**base, **fe.engineer_features(base)}
            rows.append([full[k] for k in ALL])
            labels.append(cls)
            groups.append(fname)
    return np.array(rows, dtype=np.float64), np.array(labels), np.array(groups)


class Command(BaseCommand):
    help = "Diagnostik CWRU vs PU (baca-saja)."

    def handle(self, *args, **opts):
        for names in VARIANTS.values():
            for n in names:
                if n not in ALL:
                    raise CommandError("fitur tidak dikenal: " + n)

        print("Memuat dan mengekstrak fitur CWRU (39 file)...")
        Xc, yc, gc = load_cwru()
        print("CWRU: %d jendela, %d file" % (len(yc), len(set(gc.tolist()))))

        d = np.load(CACHE, allow_pickle=True)
        keep = d["rpm_sane"]
        yp = d["label"][keep]
        op = d["origin"][keep]
        rows = []
        for r in d["features"][keep]:
            base = dict(zip(BASE, map(float, r)))
            full = {**base, **fe.engineer_features(base)}
            rows.append([full[k] for k in ALL])
        Xp = np.array(rows, dtype=np.float64)
        Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
        is_h, is_a, is_r = op == "healthy", op == "artificial", op == "real"
        print("PU: %d jendela" % len(yp))

        # ---------------- A. sebaran fitur CWRU vs PU ----------------
        print("\n================ A. p50 [p5 - p95]: CWRU vs PU ================")
        groups = [
            ("CWRU HEALTHY", Xc[yc == "HEALTHY"]), ("PU   HEALTHY", Xp[is_h]),
            ("CWRU OUTER", Xc[yc == "OUTER_RACE"]),
            ("PU   OUTER buatan", Xp[is_a & (yp == "OUTER_RACE")]),
            ("PU   OUTER nyata", Xp[is_r & (yp == "OUTER_RACE")]),
            ("CWRU INNER", Xc[yc == "INNER_RACE"]),
            ("PU   INNER buatan", Xp[is_a & (yp == "INNER_RACE")]),
            ("PU   INNER nyata", Xp[is_r & (yp == "INNER_RACE")]),
            ("CWRU BALL", Xc[yc == "BALL"]),
        ]
        summary = {"A_percentiles": {}, "A_out_of_range": {}, "B": {}}
        for name in BASE:
            j = ALL.index(name)
            print("\n[%s]" % name)
            summary["A_percentiles"][name] = {}
            for g, arr in groups:
                print("  %-18s %s" % (g, pstat(arr[:, j])))
                summary["A_percentiles"][name][g] = [float(np.percentile(arr[:, j], q)) for q in (5, 50, 95)]
            mc, mp = np.median(Xc[yc == "HEALTHY"][:, j]), np.median(Xp[is_h][:, j])
            if mc != 0:
                print("  rasio median PU-sehat / CWRU-sehat = %.3g" % (mp / mc))

        print("\n--- Fraksi jendela PU di LUAR [min,max] seluruh CWRU, per fitur ---")
        print("%-22s %10s %10s" % ("fitur", "semua PU", "PU sehat"))
        for j, name in enumerate(ALL):
            lo, hi = Xc[:, j].min(), Xc[:, j].max()
            out = (Xp[:, j] < lo) | (Xp[:, j] > hi)
            print("%-22s %10.3f %10.3f" % (name, out.mean(), out[is_h].mean()))
            summary["A_out_of_range"][name] = [float(out.mean()), float(out[is_h].mean())]

        # ---------------- B. ablasi fitur bebas-skala (DIAGNOSTIK) ----------------
        print("\n================ B. ABLASI (diagnostik, BUKAN bukti zero-shot) ================")
        masks = {"buatan": is_h | is_a, "nyata": is_h | is_r}
        for k, m in masks.items():
            y = yp[m]
            maj = max(np.sum(y == t) for t in LAB3) / len(y)
            print("kelas mayoritas %s = %.4f | recall-rata-rata chance = 0.333" % (k, maj))
        hdr = "%-17s %-9s %7s | %-27s | %-45s"
        print(hdr % ("varian", "latih", "CV-cwru", "BUATAN acc / mrecall", "NYATA acc / mrecall / recall H,O,I"))
        for vname, cols in VARIANTS.items():
            idx = [ALL.index(c) for c in cols]
            for drop_ball in (False, True):
                m = (yc != "BALL") if drop_ball else np.ones(len(yc), bool)
                Xt, yt, gt = Xc[m][:, idx], yc[m], gc[m]
                sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                accs = []
                for tr, te in sg.split(Xt, yt, gt):
                    c = mk()
                    c.fit(Xt[tr], yt[tr])
                    accs.append(float((c.predict(Xt[te]) == yt[te]).mean()))
                clf = mk()
                clf.fit(Xt, yt)
                pred = clf.predict(Xp[:, idx])
                res = {}
                for k, mm in masks.items():
                    y, p = yp[mm], pred[mm]
                    rec = [float((p[y == t] == t).mean()) for t in LAB3]
                    res[k] = {"acc": float((y == p).mean()), "mrecall": float(np.mean(rec)), "recall": rec}
                label = "3 kelas" if drop_ball else "4 kelas"
                a, r = res["buatan"], res["nyata"]
                print("%-17s %-9s %7.4f | %6.4f / %6.4f%12s | %6.4f / %6.4f / %.2f, %.2f, %.2f" % (
                    vname, label, float(np.mean(accs)), a["acc"], a["mrecall"], "",
                    r["acc"], r["mrecall"], r["recall"][0], r["recall"][1], r["recall"][2]))
                summary["B"][vname + "|" + label] = {"cv_cwru": float(np.mean(accs)), **res}

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "diag_cwru_vs_pu_g3b.json"
        out.write_text(json.dumps(summary, indent=2))
        print("\nJSON ringkasan: %s" % out)
