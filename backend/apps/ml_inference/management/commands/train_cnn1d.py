"""Sesi D G4: 1D-CNN (cnn1d-raw) pada CWRU + evaluasi zero-shot PU.

KONFIGURASI TERKUNCI sebelum ada hasil CNN (lihat CONFIG): tidak ada
hyperparameter yang diubah setelah melihat hasil PU. Command ini:
  1. CV 5-fold (fold IDENTIK dengan training rf-baseline) pada 39 file CWRU, 4 kelas;
  2. melatih model final (seed 42/43/44) pada seluruh file CWRU, evaluasi zero-shot PU;
  3. satu varian DIAGNOSTIK 3 kelas (tanpa BALL), seed 42, final-fit saja;
  4. menyimpan artefak seed 42 + eval_results JSON. TIDAK menyentuh registry (itu G5).
Gunakan --smoke dulu (1 epoch, 1 fold, tidak menyimpan apa pun).
"""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal as sps
import torch
import torch.nn as nn
from django.core.management.base import BaseCommand, CommandError
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from apps.ml_inference import pu_dataset as pud
from apps.ml_inference.management.commands import train_rf_baseline as trb

CONFIG = {
    "input": "jendela 1,0 s -> pu_dataset.to_cnn_window -> 12000 sampel -> z-score per jendela (eps 1e-8)",
    "augment": "overlap 75% (hop 0,25 s) hanya pada file latih; jendela uji non-overlap (= 342 jendela RF)",
    "arch": "Conv(1,16,k64,s16)+Conv(16,32,k3,p1)+Conv(32,64,k3,p1)+Conv(64,64,k3,p1); tiap blok BN+ReLU+MaxPool2; GAP; Dropout0.5; Linear(64,K)",
    "optimizer": "Adam lr=1e-3 weight_decay=1e-4",
    "batch": 64, "epochs": 30, "loss": "CrossEntropy, bobot kelas seimbang (dari jendela latih)",
    "cv": "StratifiedGroupKFold(5, shuffle=True, random_state=42) pada urutan jendela RF; seed model 42",
    "pu_seeds": [42, 43, 44], "threshold": 0.6,
}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
CLASSES4 = ["BALL", "HEALTHY", "INNER_RACE", "OUTER_RACE"]
CLASSES3 = ["HEALTHY", "INNER_RACE", "OUTER_RACE"]
LAB3 = ["HEALTHY", "OUTER_RACE", "INNER_RACE"]
PU_CACHE = Path("/app/ml_artifacts/pu_cache")
REGISTRY = Path("/app/config/model_registry.json")
OUT_DIR = Path("/app/ml_artifacts/cnn1d-eval")
ART_DIR = Path("/app/ml_artifacts/cnn1d-cwru-v1.0")
THRESH = CONFIG["threshold"]


class Cnn1d(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def blk(cin, cout, k, s=1, p=0):
            return nn.Sequential(nn.Conv1d(cin, cout, k, stride=s, padding=p),
                                 nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(blk(1, 16, 64, 16, 0), blk(16, 32, 3, 1, 1),
                                      blk(32, 64, 3, 1, 1), blk(64, 64, 3, 1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc(self.drop(self.features(x).mean(dim=2)))


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def train(X, y_idx, n_classes, seed, epochs):
    torch.manual_seed(seed)
    model = Cnn1d(n_classes)
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    w = len(y_idx) / (n_classes * np.maximum(counts, 1.0))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(np.ascontiguousarray(X[:, None, :]))
    yt = torch.from_numpy(y_idx).long()
    n = len(yt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, CONFIG["batch"]):
            b = perm[i:i + CONFIG["batch"]]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(Xt[b]), yt[b]).backward()
            opt.step()
    return model


def predict_proba(model, X, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs][:, None, :]))
            out.append(torch.softmax(model(xb), dim=1).numpy())
    return np.concatenate(out).astype(np.float64)


class Command(BaseCommand):
    help = "G4: 1D-CNN CWRU + evaluasi zero-shot PU."

    def add_arguments(self, parser):
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--force", action="store_true")

    def handle(self, *args, **opts):
        smoke = opts["smoke"]
        epochs = 1 if smoke else CONFIG["epochs"]
        seeds = [42] if smoke else CONFIG["pu_seeds"]
        if ART_DIR.exists() and not opts["force"] and not smoke:
            raise CommandError("artefak sudah ada: " + str(ART_DIR) + " (pakai --force untuk menimpa)")
        print("CONFIG_HASH:", CONFIG_HASH)
        print(json.dumps(CONFIG, indent=1))

        # ---- cek preprocessing ----
        probe = np.random.default_rng(0).standard_normal(trb.WINDOW_SAMPLES)
        chk = np.asarray(pud.to_cnn_window(probe, fs_hz=trb.SAMPLE_RATE_HZ))
        if chk.shape != (pud.CNN_WINDOW_SAMPLES,):
            raise CommandError("to_cnn_window(48 kHz) bentuk keluaran salah: " + str(chk.shape))
        ref = sps.resample(probe, pud.CNN_WINDOW_SAMPLES)
        print("cek to_cnn_window vs scipy.signal.resample: maks selisih = %.3g" % np.abs(chk - ref).max())

        # ---- CWRU ----
        waves, hashes = {}, {}
        for fname in sorted(trb.FILE_MANIFEST):
            mat_id = trb.FILE_MANIFEST[fname][0]
            path = trb.DATASET_ROOT / fname
            hashes[fname] = trb._sha256_file(path)
            key = "X%s_DE_time" % mat_id
            waves[fname] = sio.loadmat(path, variable_names=[key])[key].squeeze().astype(np.float64)
        ds_hash = hashlib.sha256("\n".join("%s:%s" % (f, hashes[f]) for f in sorted(hashes)).encode()).hexdigest()
        W = trb.WINDOW_SAMPLES
        HOP = W // 4
        rows, ys, gs, nonov, y_rf, g_rf = [], [], [], [], [], []
        for fname in sorted(waves):
            cls = trb.FILE_MANIFEST[fname][1]
            wf = waves[fname]
            n = len(wf) // W
            for _w in range(n):
                y_rf.append(cls)
                g_rf.append(fname)
            for s in range(0, len(wf) - W + 1, HOP):
                rows.append(np.asarray(pud.to_cnn_window(wf[s:s + W], fs_hz=trb.SAMPLE_RATE_HZ), dtype=np.float32))
                ys.append(cls)
                gs.append(fname)
                nonov.append(s % W == 0 and s // W < n)
        X = zscore(np.asarray(rows, dtype=np.float32))
        ys, gs, nonov = np.array(ys), np.array(gs), np.array(nonov)
        y_rf, g_rf = np.array(y_rf), np.array(g_rf)
        print("CWRU: %d jendela (augmentasi) | %d jendela non-overlap (RF: %d) | %d file" % (
            len(ys), int(nonov.sum()), len(y_rf), len(set(gs.tolist()))))
        if int(nonov.sum()) != len(y_rf):
            raise CommandError("jumlah jendela non-overlap != urutan RF")

        reg = [e for e in json.loads(REGISTRY.read_text()) if e["model_version"] == "rf-baseline-cwru-v1.1"][-1]
        print("hash dataset == hash training rf-baseline-v1.1:", ds_hash == reg.get("training_dataset_version_hash"))
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(sg.split(np.zeros((len(y_rf), 1)), y_rf, g_rf))
        try:
            same = all(sorted(set(g_rf[te].tolist())) == sorted(reg["cv_fold_details"][i]["test_groups"])
                       for i, (_, te) in enumerate(folds))
            print("fold identik dengan rf-baseline-v1.1 (test_groups):", same)
        except Exception as exc:
            print("verifikasi fold tidak bisa dilakukan:", repr(exc))
            same = None
        if smoke:
            folds = folds[:1]

        # ---- CV ----
        c2i = {c: i for i, c in enumerate(CLASSES4)}
        print("\n================ CV CWRU (4 kelas) ================")
        fold_res, cm_sum = [], np.zeros((4, 4), dtype=int)
        for fi, (_, te) in enumerate(folds, start=1):
            t0 = time.time()
            test_files = sorted(set(g_rf[te].tolist()))
            tr_m = ~np.isin(gs, test_files)
            te_m = np.isin(gs, test_files) & nonov
            ytr = np.array([c2i[c] for c in ys[tr_m]])
            model = train(X[tr_m], ytr, 4, 42, epochs)
            pr = predict_proba(model, X[te_m]).argmax(1)
            yte = np.array([c2i[c] for c in ys[te_m]])
            acc = float((pr == yte).mean())
            f1 = float(f1_score(yte, pr, average="macro"))
            cm = confusion_matrix(yte, pr, labels=range(4))
            cm_sum += cm
            fold_res.append({"fold": fi, "accuracy": acc, "macro_f1": f1, "n_train": int(tr_m.sum()),
                             "n_test": int(te_m.sum()), "test_groups": test_files})
            print("  fold %d: acc=%.4f macro_f1=%.4f (train %d, test %d) %.0f s" % (
                fi, acc, f1, tr_m.sum(), te_m.sum(), time.time() - t0))
        cv_acc = float(np.mean([r["accuracy"] for r in fold_res]))
        cv_std = float(np.std([r["accuracy"] for r in fold_res]))
        cv_f1 = float(np.mean([r["macro_f1"] for r in fold_res]))
        print("CV CNN: accuracy=%.4f (+/-%.4f) macro_f1=%.4f | rf-baseline-v1.1: accuracy=%.4f" % (
            cv_acc, cv_std, cv_f1, reg["eval_accuracy"]))
        print("konfusi teragregasi (baris=benar, kolom=prediksi; urutan %s):" % CLASSES4)
        for i, c in enumerate(CLASSES4):
            print("  %-12s %s" % (c, cm_sum[i].tolist()))

        # ---- PU ----
        d = np.load(PU_CACHE / "index.npz", allow_pickle=True)
        keep = d["rpm_sane"]
        raw = np.load(PU_CACHE / "raw12k.npy", mmap_mode="r")
        if raw.shape[0] != len(keep):
            raise CommandError("raw12k dan index.npz tidak sebaris")
        Xp = zscore(np.asarray(raw[keep], dtype=np.float32))
        yp, op, code = d["label"][keep], d["origin"][keep], d["code"][keep]
        speed = d["speed_code"][keep]
        is_h, is_a, is_r = op == "healthy", op == "artificial", op == "real"
        print("\nPU: %d jendela" % len(yp))

        def pu_report(title, classes, probs, full):
            pred = np.array(classes)[probs.argmax(1)]
            pred_thr = np.where(probs.max(1) < THRESH, "UNKNOWN", pred)
            cols = list(classes) + ["UNKNOWN"]
            res = {}
            subsets = [("SEMUA", np.ones(len(yp), bool)), ("BUATAN", is_h | is_a), ("NYATA", is_h | is_r)]
            for sp in sorted(set(speed.tolist())):
                subsets.append(("BUATAN N" + sp, (is_h | is_a) & (speed == sp)))
                subsets.append(("NYATA N" + sp, (is_h | is_r) & (speed == sp)))
            print("\n######## %s ########" % title)
            for name, m in subsets:
                y, p, pt = yp[m], pred[m], pred_thr[m]
                acc, acc_t = float((y == p).mean()), float((y == pt).mean())
                rec = {t: float((pt[y == t] == t).mean()) for t in LAB3 if (y == t).any()}
                rec_a = [float((p[y == t] == t).mean()) for t in LAB3]
                f1 = float(f1_score(y, p, labels=LAB3, average="macro", zero_division=0))
                maj = max(Counter(y.tolist()).values()) / len(y)
                res[name] = {"n": int(m.sum()), "acc": acc, "acc_thr": acc_t, "macro_f1": f1,
                             "mrecall": float(np.mean(rec_a)), "recall_argmax": rec_a, "majority": float(maj)}
                print("%-14s n=%5d acc=%.4f acc(0.6)=%.4f macroF1=%.4f mrecall=%.4f recall(H,O,I)=%.2f,%.2f,%.2f | mayoritas=%.4f" % (
                    name, m.sum(), acc, acc_t, f1, np.mean(rec_a), rec_a[0], rec_a[1], rec_a[2], maj))
                if full and name in ("BUATAN", "NYATA"):
                    print("  %-16s" % "benar/prediksi" + "".join("%12s" % c for c in cols))
                    for t in LAB3:
                        cnt = Counter(pt[y == t].tolist())
                        print("  %-16s" % t + "".join("%12d" % cnt[c] for c in cols))
            if full:
                print("\n  per kode (ambang 0,6):")
                for c in sorted(set(code.tolist())):
                    m = code == c
                    cnt = Counter(pred_thr[m].tolist())
                    print("  %-6s %-10s %-11s acc=%.3f  %s" % (
                        c, op[m][0], yp[m][0], float((pred_thr[m] == yp[m][0]).mean()),
                        ", ".join("%s:%d" % kv for kv in cnt.most_common())))
            return res, pred

        results = {"config": CONFIG, "config_hash": CONFIG_HASH, "dataset_hash": ds_hash, "smoke": smoke,
                   "cv": {"accuracy": cv_acc, "accuracy_std": cv_std, "macro_f1": cv_f1, "folds": fold_res,
                          "confusion": cm_sum.tolist(), "folds_identical_to_rf": same},
                   "rf_baseline_cv_accuracy": reg["eval_accuracy"], "pu": {}}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        final42 = None
        for sd in seeds:
            t0 = time.time()
            ytr = np.array([c2i[c] for c in ys])
            model = train(X, ytr, 4, sd, epochs)
            probs = predict_proba(model, Xp)
            res, _ = pu_report("CNN 4 kelas, seed %d (zero-shot PU)" % sd, CLASSES4, probs, full=(sd == 42))
            results["pu"]["seed%d" % sd] = res
            np.savez_compressed(OUT_DIR / ("pu_probs_seed%d%s.npz" % (sd, "_smoke" if smoke else "")),
                                probs=probs.astype(np.float32), classes=np.array(CLASSES4))
            print("  (seed %d: %.0f s)" % (sd, time.time() - t0))
            if sd == 42:
                final42 = model
        m3 = ys != "BALL"
        c3 = {c: i for i, c in enumerate(CLASSES3)}
        model3 = train(X[m3], np.array([c3[c] for c in ys[m3]]), 3, 42, epochs)
        res3, _ = pu_report("DIAGNOSTIK: CNN 3 kelas tanpa BALL, seed 42 (bukan kandidat model)",
                            CLASSES3, predict_proba(model3, Xp), full=False)
        results["pu"]["diag_3class_seed42"] = res3

        if not smoke:
            ART_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": final42.state_dict(), "classes": CLASSES4, "config": CONFIG,
                        "config_hash": CONFIG_HASH, "dataset_hash": ds_hash, "seed": 42,
                        "input_samples": pud.CNN_WINDOW_SAMPLES}, ART_DIR / "model.pt")
            print("\nartefak: %s" % (ART_DIR / "model.pt"))
        out = OUT_DIR / ("g4_results_smoke.json" if smoke else "g4_results.json")
        out.write_text(json.dumps(results, indent=2))
        print("hasil: %s" % out)
        print("\nKEPUTUSAN ml-pipeline.md section 4 (berdasarkan CV CWRU, bukan PU): CNN %.4f vs RF %.4f -> %s" % (
            cv_acc, reg["eval_accuracy"],
            "cnn1d-raw memenuhi syarat 'akurasi eval > rf-baseline'" if cv_acc > reg["eval_accuracy"]
            else "rf-baseline tetap model terdeploy"))
