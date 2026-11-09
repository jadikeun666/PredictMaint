"""
Sesi D G5: daftarkan cnn1d-cwru-v1.0 ke Model Registry (NOT_DEPLOYED) dan isi
cross_dataset_pu_accuracy rf-baseline-cwru-v1.1 (field yang null sejak 2026-09-20).

SEMUA angka dibaca dari file hasil run (g4_results.json, g4b_results.json,
rf_zero_shot_*.json, pu_probs_seed42.npz); tidak ada angka yang diketik
(engineering-rules.md). Command menolak berjalan bila: hasil smoke, config hash
G4 beda dari yang terkunci, hash dataset beda, versi sudah terdaftar, atau field
cross_dataset_pu_accuracy RF sudah terisi. Default --dry-run tidak menulis apa pun.

  python manage.py register_sesi_d            # dry-run (tampilkan saja)
  python manage.py register_sesi_d --apply    # tulis ke registry
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from apps.ml_inference import model_registry

CNN_VERSION = "cnn1d-cwru-v1.0"
RF_VERSION = "rf-baseline-cwru-v1.1"
EVAL_DIR = Path("/app/ml_artifacts/cnn1d-eval")
G4_JSON = EVAL_DIR / "g4_results.json"
G4B_JSON = EVAL_DIR / "g4b_results.json"
PROBS42 = EVAL_DIR / "pu_probs_seed42.npz"
PU_INDEX = Path("/app/ml_artifacts/pu_cache/index.npz")
RF_PU_JSON = Path("/app/ml_artifacts/pu_eval/rf_zero_shot_rf-baseline-cwru-v1.1.json")
CNN_ARTIFACT = Path("/app/ml_artifacts/cnn1d-cwru-v1.0/model.pt")
LAB3 = ("HEALTHY", "OUTER_RACE", "INNER_RACE")
CLASSES4 = ["BALL", "HEALTHY", "INNER_RACE", "OUTER_RACE"]


def g4b_summary(g4b):
    folds = g4b["folds"]
    n_tot = sum(f["n"] for f in folds)
    out = {"n_instances": len(folds), "n_windows": int(n_tot),
           "protocol": "leave-one-fault-instance-out; seluruh file satu instance (semua beban) ditahan; "
                       "HEALTHY (satu bearing) tidak bisa diuji bebas-kebocoran dan tetap di training",
           "per_instance": [{k: f[k] for k in ("instance", "class", "n", "cnn_recall", "rf_recall", "cnn_pred", "rf_pred")}
                            for f in folds]}
    for m in ("cnn", "rf"):
        out[m + "_accuracy"] = float(sum(f[m + "_recall"] * f["n"] for f in folds) / n_tot)
        out[m + "_mean_recall_per_class"] = {
            c: float(np.mean([f[m + "_recall"] for f in folds if f["class"] == c]))
            for c in sorted({f["class"] for f in folds})}
    return out


def pu_seed_summary(g4):
    seeds = sorted(k for k in g4["pu"] if k.startswith("seed"))
    per_seed = {s: {sub: {k: g4["pu"][s][sub][k] for k in ("n", "acc", "acc_thr", "macro_f1", "mrecall", "recall_argmax", "majority")}
                    for sub in ("SEMUA", "BUATAN", "NYATA")} for s in seeds}
    out = {"seeds": seeds, "per_seed": per_seed,
           "recall_order": list(LAB3),
           "protocol": "zero-shot, 3 kelas (BALL=salah), 26 kode PU, HEALTHY dipakai bersama di buatan dan nyata"}
    for sub in ("BUATAN", "NYATA", "SEMUA"):
        accs = [per_seed[s][sub]["acc"] for s in seeds]
        out[sub.lower() + "_acc_mean"] = float(np.mean(accs))
        out[sub.lower() + "_acc_std_across_seeds"] = float(np.std(accs))
        out[sub.lower() + "_majority"] = float(per_seed[seeds[0]][sub]["majority"])
    out["diag_3class_seed42"] = {sub: {k: g4["pu"]["diag_3class_seed42"][sub][k] for k in ("acc", "mrecall")}
                                 for sub in ("BUATAN", "NYATA")} if "diag_3class_seed42" in g4["pu"] else None
    return out


def confidence_summary(probs_npz, index_npz):
    p = np.load(probs_npz)
    d = np.load(index_npz, allow_pickle=True)
    y = d["label"][d["rpm_sane"]]
    probs = p["probs"].astype(np.float64)
    cls = [str(c) for c in p["classes"]]
    conf, pred = probs.max(1), np.array(cls)[probs.argmax(1)]
    out = {"model_seed": 42, "per_true_label": {}}
    for lab in LAB3:
        m = y == lab
        hi = m & (conf >= 0.85)
        out["per_true_label"][lab] = {
            "n": int(m.sum()),
            "conf_ge_085": float((conf[m] >= 0.85).mean()),
            "conf_060_085": float(((conf[m] >= 0.6) & (conf[m] < 0.85)).mean()),
            "conf_lt_060": float((conf[m] < 0.6).mean()),
            "pred_ge_085": {c: float((pred[hi] == c).mean()) for c in sorted(set(pred[hi].tolist()))} if hi.any() else {}}
    inner = (conf >= 0.85) & (pred == "INNER_RACE")
    out["pred_INNER_RACE_conf_ge_085"] = {
        "n": int(inner.sum()),
        "precision": float((y[inner] == "INNER_RACE").mean()) if inner.any() else None,
        "share_true_healthy": float((y[inner] == "HEALTHY").mean()) if inner.any() else None}
    return out


def build_rf_pu_fields(rf_pu, source_name):
    sub = rf_pu["subsets"]
    head = sub["NYATA (+healthy)"]
    return {
        "cross_dataset_pu_accuracy": float(head["acc"]),
        "cross_dataset_pu": {
            "headline": "akurasi argmax 3 kelas pada himpunan NYATA (+healthy); prediksi BALL dihitung salah",
            "protocol": "zero-shot (tanpa retraining), 26 kode PU, jendela 1,0 s, rpm_sane saja",
            "source_file": source_name,
            "subsets": sub},
        "cross_dataset_pu_note": (
            "Akurasi NYATA %.4f vs kelas mayoritas %.4f; BUATAN %.4f vs %.4f. Bukan bukti generalisasi: "
            "%s kelas mayoritas. Ambang keyakinan 0,6 memetakan sebagian jendela ke UNKNOWN (acc_thr)." % (
                head["acc"], head["majority"], sub["BUATAN (+healthy)"]["acc"], sub["BUATAN (+healthy)"]["majority"],
                "di bawah" if head["acc"] < head["majority"] else "tidak di bawah")),
    }


def build_cnn_entry(g4, g4b, rf_entry, *, artifact_path, trained_at, eval_date, confidence=None):
    cv, b, pu = g4["cv"], g4b_summary(g4b), pu_seed_summary(g4)
    notes = [
        "CV per-file terkunci (StratifiedGroupKFold 5, fold identik dengan %s): akurasi %.4f (std %.4f); pembanding RF %.4f." % (
            RF_VERSION, cv["accuracy"], cv["accuracy_std"], rf_entry["eval_accuracy"]),
        "CV per-file memisahkan FILE, bukan bearing fisik: file beban berbeda dari instance fault yang sama dapat berada di "
        "train dan test (ml-pipeline.md section 1.2 bertujuan mencegah ini). Angka CV ini bukan bukti generalisasi.",
        "Leave-one-fault-instance-out (%d instance, %d jendela fault; HEALTHY tidak diuji): CNN akurasi %.4f vs RF %.4f." % (
            b["n_instances"], b["n_windows"], b["cnn_accuracy"], b["rf_accuracy"]),
        "Zero-shot PU (3 kelas, BALL=salah), %d seed: akurasi NYATA %.4f (std antar-seed %.4f), BUATAN %.4f (std %.4f); "
        "kelas mayoritas NYATA %.4f, BUATAN %.4f -> %s kelas mayoritas." % (
            len(pu["seeds"]), pu["nyata_acc_mean"], pu["nyata_acc_std_across_seeds"], pu["buatan_acc_mean"],
            pu["buatan_acc_std_across_seeds"], pu["nyata_majority"], pu["buatan_majority"],
            "di bawah" if max(pu["nyata_acc_mean"] - pu["nyata_majority"], pu["buatan_acc_mean"] - pu["buatan_majority"]) < 0
            else "tidak seluruhnya di bawah"),
        "Perilaku di PU tidak stabil antar-seed: recall INNER_RACE (semua PU, argmax) berkisar %.2f-%.2f." % (
            min(pu["per_seed"][s]["SEMUA"]["recall_argmax"][2] for s in pu["seeds"]),
            max(pu["per_seed"][s]["SEMUA"]["recall_argmax"][2] for s in pu["seeds"])),
        "Jalur inference torch (.pt) belum ada di apps.ml_inference.tasks (hanya joblib); model TIDAK dideploy.",
    ]
    entry = {
        "model_version": CNN_VERSION, "model_type": "CLASSIFICATION", "training_dataset": "CWRU",
        "training_dataset_version_hash": g4["dataset_hash"],
        "eval_accuracy": float(cv["accuracy"]), "eval_accuracy_std": float(cv["accuracy_std"]),
        "eval_macro_f1": float(cv["macro_f1"]),
        "eval_protocol": "CV per-file 5 fold (fold identik dengan rf-baseline), seed model 42; pelengkap: leave-instance-out (G4b)",
        "eval_date": eval_date, "trained_at": trained_at,
        "artifact_path": str(artifact_path), "artifact_format": "torch_state_dict (model.pt; payload: state_dict, classes, config)",
        "label_classes": CLASSES4,
        "input_spec": g4["config"]["input"],
        "hyperparameters": g4["config"], "config_hash": g4["config_hash"],
        "cv_fold_details": cv["folds"], "cv_confusion": cv["confusion"], "cv_confusion_order": CLASSES4,
        "eval_leave_instance_out": b,
        "cross_dataset_pu_accuracy": pu["nyata_acc_mean"],
        "cross_dataset_pu_accuracy_basis": "rata-rata %d seed, himpunan NYATA (+healthy), argmax 3 kelas, BALL=salah" % len(pu["seeds"]),
        "cross_dataset_pu": pu,
        "n_training_windows_augmented": None, "n_training_files": 39,
        "deployment": "NOT_DEPLOYED",
        "deployment_note": (
            "Memenuhi syarat formal ml-pipeline.md section 4 (akurasi CV CWRU %.4f > %.4f) tetapi TIDAK dideploy: "
            "jalur inference torch belum ada, dan hasil leave-instance-out serta PU tidak mendukung generalisasi. "
            "get_latest_entry('CLASSIFICATION') melewati entri NOT_DEPLOYED." % (cv["accuracy"], rf_entry["eval_accuracy"])),
        "evaluation_notes": notes, "supersedes": None,
    }
    if confidence is not None:
        entry["cross_dataset_pu_confidence_seed42"] = confidence
    return entry


class Command(BaseCommand):
    help = "G5: registrasi cnn1d-cwru-v1.0 (NOT_DEPLOYED) + isi cross_dataset_pu_accuracy rf-baseline-v1.1."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true", help="tulis ke registry (default: dry-run)")

    def handle(self, *args, **o):
        from apps.ml_inference.management.commands import train_cnn1d as g4mod  # impor torch hanya di sini

        for pth in (G4_JSON, G4B_JSON, RF_PU_JSON, CNN_ARTIFACT):
            if not pth.exists():
                raise CommandError("tidak ada: %s" % pth)
        g4 = json.loads(G4_JSON.read_text())
        g4b = json.loads(G4B_JSON.read_text())
        rf_pu = json.loads(RF_PU_JSON.read_text())
        if g4.get("smoke") or g4b.get("smoke"):
            raise CommandError("hasil smoke tidak boleh diregistrasi")
        if g4["config_hash"] != g4mod.CONFIG_HASH or g4b["config_hash"] != g4mod.CONFIG_HASH:
            raise CommandError("config hash hasil != CONFIG terkunci saat ini")
        reg = model_registry.load_registry()
        if any(e.get("model_version") == CNN_VERSION for e in reg):
            raise CommandError(CNN_VERSION + " sudah terdaftar")
        rf = [e for e in reg if e.get("model_version") == RF_VERSION]
        if len(rf) != 1:
            raise CommandError("entri %s harus tepat satu, ditemukan %d" % (RF_VERSION, len(rf)))
        rf = rf[0]
        if rf.get("cross_dataset_pu_accuracy") is not None:
            raise CommandError("cross_dataset_pu_accuracy RF sudah terisi; tidak ditimpa")
        if rf["training_dataset_version_hash"] != g4["dataset_hash"]:
            raise CommandError("hash dataset RF != hash dataset CNN")

        conf = confidence_summary(PROBS42, PU_INDEX)
        entry = build_cnn_entry(
            g4, g4b, rf, artifact_path=CNN_ARTIFACT,
            trained_at=datetime.fromtimestamp(os.path.getmtime(CNN_ARTIFACT), timezone.utc).isoformat(),
            eval_date=datetime.now(timezone.utc).date().isoformat(), confidence=conf)
        rf_fields = build_rf_pu_fields(rf_pu, RF_PU_JSON.name)
        missing = [f for f in model_registry.REQUIRED_FIELDS if f not in entry]
        if missing:
            raise CommandError("entri CNN kurang field: %s" % missing)
        after = reg + [entry]
        winner = max([e for e in after if e.get("model_type") == "CLASSIFICATION" and model_registry._is_deployable(e)],
                     key=lambda e: e["eval_date"])
        if winner["model_version"] != RF_VERSION:
            raise CommandError("setelah registrasi model terdeploy akan menjadi %s, bukan %s" % (
                winner["model_version"], RF_VERSION))

        self.stdout.write("== RF: field yang akan diisi ==")
        self.stdout.write("cross_dataset_pu_accuracy = %.4f" % rf_fields["cross_dataset_pu_accuracy"])
        self.stdout.write(rf_fields["cross_dataset_pu_note"])
        self.stdout.write("\n== CNN: entri ringkas ==")
        for k in ("model_version", "eval_accuracy", "eval_accuracy_std", "eval_macro_f1", "cross_dataset_pu_accuracy",
                  "deployment", "artifact_path"):
            self.stdout.write("%s = %s" % (k, entry[k]))
        self.stdout.write("leave-instance-out: CNN %.4f | RF %.4f" % (
            entry["eval_leave_instance_out"]["cnn_accuracy"], entry["eval_leave_instance_out"]["rf_accuracy"]))
        self.stdout.write("konfiden PU (seed 42): %s" % json.dumps(conf["pred_INNER_RACE_conf_ge_085"]))
        for n in entry["evaluation_notes"]:
            self.stdout.write("- " + n)
        self.stdout.write("model terdeploy sesudah registrasi: %s" % winner["model_version"])
        if not o["apply"]:
            self.stdout.write("\n(dry-run; tidak ada yang ditulis. Ulangi dengan --apply)")
            return
        model_registry.append_entry(entry)
        model_registry.update_entry_fields(RF_VERSION, rf_fields)
        latest = model_registry.get_latest_entry("CLASSIFICATION")
        self.stdout.write("\nTERTULIS. get_latest_entry('CLASSIFICATION') -> %s" % latest["model_version"])
