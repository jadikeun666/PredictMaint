"""Sesi D G3: evaluasi zero-shot rf-baseline (CWRU) pada cache PU. BACA-SAJA.

Tidak mengubah registry/model/cache. Keluaran: laporan ke stdout + JSON ringkasan
di /app/ml_artifacts/pu_eval/. Evaluasi 3 kelas (HEALTHY/OUTER_RACE/INNER_RACE);
prediksi BALL dihitung sebagai salah. Hasil dipisah asal buatan vs nyata
(HEALTHY dipakai sebagai referensi bersama di kedua himpunan).
"""
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from django.core.management.base import BaseCommand, CommandError
from sklearn.metrics import f1_score

from apps.ml_inference import feature_extraction as fe
from apps.ml_inference import features_generic as fg

CACHE = Path("/app/ml_artifacts/pu_cache/index.npz")
REGISTRY = Path("/app/config/model_registry.json")
OUT_DIR = Path("/app/ml_artifacts/pu_eval")
LAB = ["HEALTHY", "OUTER_RACE", "INNER_RACE"]
THRESH = 0.6  # ml-pipeline.md §1.5


def _pct(a, q):
    return float(np.percentile(a, q))


class Command(BaseCommand):
    help = "Evaluasi zero-shot rf-baseline pada PU (baca-saja)."

    def add_arguments(self, parser):
        parser.add_argument("--model", default="rf-baseline-cwru-v1.1")

    def handle(self, *args, **opts):
        entries = json.loads(REGISTRY.read_text())
        entries = entries if isinstance(entries, list) else list(entries.get("models", entries).values())
        match = [e for e in entries if e.get("model_version") == opts["model"]]
        if not match:
            raise CommandError("model tidak ada di registry")
        entry = match[-1]
        obj = joblib.load(entry["artifact_path"])
        clf = obj if hasattr(obj, "predict_proba") else next(
            v for v in obj.values() if hasattr(v, "predict_proba"))
        classes = [str(c) for c in clf.classes_]
        if classes != list(entry["label_classes"]):
            raise CommandError(f"urutan kelas beda: {classes} vs registry")

        d = np.load(CACHE, allow_pickle=True)
        keep = d["rpm_sane"]
        F = d["features"][keep]
        true = d["label"][keep]
        origin = d["origin"][keep]
        code = d["code"][keep]
        speed = d["speed_code"][keep]
        torque = d["torque_code"][keep]
        force = d["force_code"][keep]
        fidx = d["file_index"][keep]
        print(f"jendela dipakai: {keep.sum()} / {len(keep)} (rpm_sane)")

        base_cols = list(fe.FEATURE_ORDER)
        if F.shape[1] != len(base_cols):
            raise CommandError("jumlah kolom cache != FEATURE_ORDER")
        probe = {k: float(i + 1) for i, k in enumerate(base_cols)}
        if not np.allclose(fg.to_vector(probe), np.arange(1, len(base_cols) + 1)):
            raise CommandError("urutan kolom to_vector != FEATURE_ORDER")

        rows = []
        for r in F:
            base = dict(zip(base_cols, map(float, r)))
            merged = {**base, **fe.engineer_features(base)}
            rows.append([merged[k] for k in entry["feature_order"]])
        X = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(X).all():
            print("PERINGATAN: ada nilai non-finite:", (~np.isfinite(X)).sum())
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        proba = clf.predict_proba(X)
        pred = np.array(classes)[proba.argmax(1)]
        conf = proba.max(1)
        pred_thr = np.where(conf < THRESH, "UNKNOWN", pred)
        cols = classes + ["UNKNOWN"]
        result = {"model": entry["model_version"], "subsets": {}}

        def summarize(mask, name):
            y, p, pt = true[mask], pred[mask], pred_thr[mask]
            n = int(mask.sum())
            if n == 0:
                return
            nfiles = len(set(zip(code[mask].tolist(), fidx[mask].tolist())))
            acc, acc_t = float((y == p).mean()), float((y == pt).mean())
            f1 = float(f1_score(y, p, labels=LAB, average="macro", zero_division=0))
            maj = max(Counter(y.tolist()).values()) / n
            print(f"\n--- {name} | n={n} jendela, {nfiles} file ---")
            print(f"akurasi argmax={acc:.4f} | akurasi dgn ambang 0,6 (UNKNOWN=salah)={acc_t:.4f} "
                  f"| macro-F1(3 kelas)={f1:.4f} | kelas mayoritas={maj:.4f}")
            print("benar/prediksi".ljust(18) + "".join(c.rjust(12) for c in cols) + "recall".rjust(9))
            rec = {}
            for t in LAB:
                m = y == t
                if not m.any():
                    continue
                cnt = Counter(pt[m].tolist())
                rc = cnt[t] / m.sum()
                rec[t] = float(rc)
                print(f"{t:<18}" + "".join(f"{cnt[c]:>12}" for c in cols) + f"{rc:>9.3f}")
            result["subsets"][name] = {"n": n, "files": nfiles, "acc": acc, "acc_thr": acc_t,
                                       "macro_f1": f1, "majority": float(maj), "recall": rec}

        is_h = origin == "healthy"
        is_a = origin == "artificial"
        is_r = origin == "real"
        print("\n================ RINGKAS PER ASAL ================")
        summarize(np.ones(len(true), bool), "SEMUA")
        summarize(is_h | is_a, "BUATAN (+healthy)")
        summarize(is_h | is_r, "NYATA (+healthy)")
        print("\n================ PER KONDISI (kode kecepatan) ================")
        for sp in sorted(set(speed.tolist())):
            summarize((is_h | is_a) & (speed == sp), f"BUATAN N{sp}")
            summarize((is_h | is_r) & (speed == sp), f"NYATA N{sp}")

        print("\n================ PER KODE BEARING ================")
        print(f"{'kode':<7}{'asal':<11}{'label':<12}{'n':>6}{'acc':>8}  distribusi prediksi (ambang 0,6)")
        for c in sorted(set(code.tolist())):
            m = code == c
            o, lb = origin[m][0], true[m][0]
            cnt = Counter(pred_thr[m].tolist())
            dist = ", ".join(f"{k}:{v}" for k, v in cnt.most_common())
            print(f"{c:<7}{o:<11}{lb:<12}{int(m.sum()):>6}{float((pred_thr[m]==lb).mean()):>8.3f}  {dist}")

        print("\n================ DIAGNOSIS SKALA: p50 [p5-p95] per grup ================")
        names = base_cols
        groups = [("HEALTHY", is_h), ("OUTER buatan", is_a & (true == "OUTER_RACE")),
                  ("OUTER nyata", is_r & (true == "OUTER_RACE")),
                  ("INNER buatan", is_a & (true == "INNER_RACE")),
                  ("INNER nyata", is_r & (true == "INNER_RACE"))]
        diag = {}
        for j, nm in enumerate(names):
            print(f"\n[{nm}]")
            for g, m in groups:
                a = X[m][:, entry["feature_order"].index(nm)]
                print(f"  {g:<14} {_pct(a,50):>12.4g}  [{_pct(a,5):.4g} - {_pct(a,95):.4g}]")
                diag.setdefault(nm, {})[g] = [_pct(a, 5), _pct(a, 50), _pct(a, 95)]
        result["scale_diagnosis"] = diag

        print("\n================ MEDIAN RMS / KURTOSIS per kondisi: SEHAT vs RUSAK ================")
        ri = entry["feature_order"].index("rms")
        ki = entry["feature_order"].index("kurtosis")
        for lab, arr in (("kecepatan", speed), ("torsi", torque), ("gaya radial", force)):
            for v in sorted(set(arr.tolist())):
                mh = is_h & (arr == v)
                mf = (~is_h) & (arr == v)
                if mh.any() and mf.any():
                    print(f"{lab} {v}: SEHAT n={int(mh.sum()):>5} rms={np.median(X[mh,ri]):.4g} kurt={np.median(X[mh,ki]):.4g}"
                          f" | RUSAK n={int(mf.sum()):>5} rms={np.median(X[mf,ri]):.4g} kurt={np.median(X[mf,ki]):.4g}")
        print("\nRMS median per kode SEHAT:")
        for c in sorted(set(code[is_h].tolist())):
            m = code == c
            print(f"  {c}: rms={np.median(X[m,ri]):.4g} kurt={np.median(X[m,ki]):.4g} n={int(m.sum())}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"rf_zero_shot_{entry['model_version']}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"\nJSON ringkasan: {out}")
