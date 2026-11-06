"""Sesi D G4b: CV PELENGKAP bebas-kebocoran (leave-one-fault-instance-out) pada CWRU.

Tiap fold menahan SELURUH file satu instance fault (mis. B007 pada semua beban
0-3), sehingga bearing fisik+kerusakan yang sama tidak ada di train dan test
(ml-pipeline.md section 1.2: tujuan split). HEALTHY (satu bearing, 4 beban) tidak
bisa diuji bebas-kebocoran; ia tetap di training pada semua fold = limitasi.
Ini PELENGKAP; tidak menggantikan CV per-file terkunci (86,22% RF / 99,38% CNN).
RF: fitur+hyperparameter v1.1 dari registry. CNN: CONFIG G4 terkunci (hash dicek).
BACA-SAJA: tidak menyentuh registry/artefak. Gunakan --smoke (1 epoch, 2 fold) dulu.
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
from django.core.management.base import BaseCommand, CommandError
from sklearn.ensemble import RandomForestClassifier

from apps.ml_inference import feature_extraction as fe
from apps.ml_inference import pu_dataset as pud
from apps.ml_inference.management.commands import train_cnn1d as g4
from apps.ml_inference.management.commands import train_rf_baseline as trb

LOCKED_HASH = "5be663c890d5644a7c163feb267d58e836903b08f9e97b05d25b5ae9797d9942"
REGISTRY = Path("/app/config/model_registry.json")
OUT = Path("/app/ml_artifacts/cnn1d-eval/g4b_results.json")
ALL = list(fe.MODEL_FEATURE_ORDER)


def instance_of(fname):
    return "HEALTHY" if fname.startswith("Time_Normal") else fname.split("_")[0]


def dist(arr):
    return ", ".join("%s:%d" % kv for kv in Counter(arr.tolist()).most_common())


class Command(BaseCommand):
    help = "G4b: CV leave-one-fault-instance-out (pelengkap, baca-saja)."

    def add_arguments(self, parser):
        parser.add_argument("--smoke", action="store_true")

    def handle(self, *args, **opts):
        smoke = opts["smoke"]
        if g4.CONFIG_HASH != LOCKED_HASH:
            raise CommandError("CONFIG G4 berbeda dari yang terkunci: " + g4.CONFIG_HASH)
        epochs = 1 if smoke else g4.CONFIG["epochs"]
        reg = [e for e in json.loads(REGISTRY.read_text()) if e["model_version"] == "rf-baseline-cwru-v1.1"][-1]
        hp = reg["hyperparameters"]
        print("CONFIG_HASH G4 cocok dengan kunci. RF hyperparameter:", hp)

        W = trb.WINDOW_SAMPLES
        HOP = W // 4
        rows, ys, gs, nonov = [], [], [], []
        xrf, yrf, grf = [], [], []
        for fname in sorted(trb.FILE_MANIFEST):
            mat_id, cls, load = trb.FILE_MANIFEST[fname]
            key = "X%s_DE_time" % mat_id
            mat = sio.loadmat(trb.DATASET_ROOT / fname, variable_names=[key, "X%sRPM" % mat_id])
            wf = mat[key].squeeze().astype(np.float64)
            rpm = trb._load_rpm(mat, mat_id, load, fname)
            inst = instance_of(fname)
            n = len(wf) // W
            for s in range(0, len(wf) - W + 1, HOP):
                chunk = wf[s:s + W]
                rows.append(np.asarray(pud.to_cnn_window(chunk, fs_hz=trb.SAMPLE_RATE_HZ), dtype=np.float32))
                ys.append(cls)
                gs.append(inst)
                is_non = (s % W == 0 and s // W < n)
                nonov.append(is_non)
                if is_non:
                    base = fe.extract_feature_vector(chunk, trb.SAMPLE_RATE_HZ, rpm)
                    full = {**base, **fe.engineer_features(base)}
                    xrf.append([full[k] for k in ALL])
                    yrf.append(cls)
                    grf.append(inst)
        X = g4.zscore(np.asarray(rows, dtype=np.float32))
        ys, gs, nonov = np.array(ys), np.array(gs), np.array(nonov)
        xrf, yrf, grf = np.array(xrf, dtype=np.float64), np.array(yrf), np.array(grf)
        if int(nonov.sum()) != len(yrf) or len(yrf) != 342:
            raise CommandError("jumlah jendela non-overlap tidak 342")
        c2i = {c: i for i, c in enumerate(g4.CLASSES4)}
        insts = sorted(set(grf.tolist()) - {"HEALTHY"})
        print("instance fault (%d): %s" % (len(insts), insts))
        if smoke:
            insts = insts[:2]

        print("\n%-8s %-11s %4s | %-8s %-34s | %-8s %s" % (
            "instance", "kelas", "n", "CNN rec", "CNN prediksi", "RF rec", "RF prediksi"))
        res = []
        for inst in insts:
            t0 = time.time()
            te_c = nonov & (gs == inst)
            tr_c = gs != inst
            true_cls = ys[te_c][0]
            model = g4.train(X[tr_c], np.array([c2i[c] for c in ys[tr_c]]), 4, 42, epochs)
            pc = np.array(g4.CLASSES4)[g4.predict_proba(model, X[te_c]).argmax(1)]
            te_r = grf == inst
            if int(te_c.sum()) != int(te_r.sum()):
                raise CommandError("jumlah jendela uji CNN != RF untuk " + inst)
            clf = RandomForestClassifier(random_state=42, class_weight="balanced", **hp)
            clf.fit(xrf[~te_r], yrf[~te_r])
            pr = clf.predict(xrf[te_r])
            r_c, r_r = float((pc == true_cls).mean()), float((pr == true_cls).mean())
            res.append({"instance": inst, "class": str(true_cls), "n": int(te_c.sum()),
                        "cnn_recall": r_c, "rf_recall": r_r,
                        "cnn_pred": dict(Counter(pc.tolist())), "rf_pred": dict(Counter(pr.tolist()))})
            print("%-8s %-11s %4d | %-8.3f %-34s | %-8.3f %s   (%.0f s)" % (
                inst, true_cls, te_c.sum(), r_c, dist(pc), r_r, dist(pr), time.time() - t0))

        n_tot = sum(r["n"] for r in res)
        print("\n================ RINGKASAN (hanya window fault; HEALTHY tidak bisa diuji bebas-kebocoran) ================")
        for name, k in (("CNN", "cnn_recall"), ("RF ", "rf_recall")):
            acc = sum(r[k] * r["n"] for r in res) / n_tot
            per_inst = float(np.mean([r[k] for r in res]))
            per_cls = {c: float(np.mean([r[k] for r in res if r["class"] == c]))
                       for c in ("BALL", "INNER_RACE", "OUTER_RACE") if any(r["class"] == c for r in res)}
            print("%s: akurasi (berbobot jendela, n=%d)=%.4f | rata-rata recall per instance=%.4f | per kelas=%s" % (
                name, n_tot, acc, per_inst, {c: round(v, 3) for c, v in per_cls.items()}))
        print("pembanding CV per-file terkunci: CNN 0.9938, RF 0.8622 (kedua angka itu mencakup HEALTHY)")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out = OUT.with_name("g4b_results_smoke.json") if smoke else OUT
        out.write_text(json.dumps({"config_hash": g4.CONFIG_HASH, "smoke": smoke, "folds": res}, indent=2))
        print("hasil: %s" % out)
