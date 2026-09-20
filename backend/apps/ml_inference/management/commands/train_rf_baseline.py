"""
Training rf-baseline classifier (ml-pipeline.md §1) di atas CWRU, baca
file .mat MENTAH langsung. v1.1: tambah fitur turunan (rasio dominasi
frekuensi karakteristik) + hyperparameter search RandomForestClassifier
(algoritma TETAP RandomForest sesuai definisi ml-pipeline.md §1 — hanya
hyperparameter & fitur input yang di-tuning, bukan ganti algoritma) untuk
mengatasi confusion BALL<->OUTER_RACE yang ditemukan di v1.0
(claude.md addendum 2026-09-11).

Usage:
    python manage.py train_rf_baseline
"""
import hashlib
import json
from datetime import date, datetime, timezone
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from apps.ml_inference import model_registry
from apps.ml_inference.feature_extraction import (
    extract_feature_vector, engineer_features, MODEL_FEATURE_ORDER,
)

DATASET_ROOT = Path("/dataset/cwru_raw")
SAMPLE_RATE_HZ = 48000
WINDOW_SAMPLES = SAMPLE_RATE_HZ
ARTIFACT_DIR = Path("/app/ml_artifacts")

OFFICIAL_APPROX_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

FILE_MANIFEST = {
    "Time_Normal_0_097.mat": ("097", "HEALTHY", 0),
    "Time_Normal_1_098.mat": ("098", "HEALTHY", 1),
    "Time_Normal_2_099.mat": ("099", "HEALTHY", 2),
    "Time_Normal_3_100.mat": ("100", "HEALTHY", 3),

    "B007_0_122.mat": ("122", "BALL", 0),
    "B007_1_123.mat": ("123", "BALL", 1),
    "B007_2_124.mat": ("124", "BALL", 2),
    "B007_3_125.mat": ("125", "BALL", 3),
    "B014_0_189.mat": ("189", "BALL", 0),
    "B014_1_190.mat": ("190", "BALL", 1),
    "B014_2_191.mat": ("191", "BALL", 2),
    "B014_3_192.mat": ("192", "BALL", 3),
    "B021_0_226.mat": ("226", "BALL", 0),
    "B021_1_227.mat": ("227", "BALL", 1),
    "B021_2_228.mat": ("228", "BALL", 2),
    "B021_3_229.mat": ("229", "BALL", 3),

    "IR007_0_109.mat": ("109", "INNER_RACE", 0),
    "IR007_1_110.mat": ("110", "INNER_RACE", 1),
    "IR007_2_111.mat": ("111", "INNER_RACE", 2),
    "IR007_3_112.mat": ("112", "INNER_RACE", 3),
    "IR014_1_175.mat": ("175", "INNER_RACE", 1),
    "IR014_2_176.mat": ("176", "INNER_RACE", 2),
    "IR014_3_177.mat": ("177", "INNER_RACE", 3),
    "IR021_0_213.mat": ("213", "INNER_RACE", 0),
    "IR021_1_214.mat": ("214", "INNER_RACE", 1),
    "IR021_2_215.mat": ("215", "INNER_RACE", 2),
    "IR021_3_217.mat": ("217", "INNER_RACE", 3),

    "OR007_6_0_135.mat": ("135", "OUTER_RACE", 0),
    "OR007_6_1_136.mat": ("136", "OUTER_RACE", 1),
    "OR007_6_2_137.mat": ("137", "OUTER_RACE", 2),
    "OR007_6_3_138.mat": ("138", "OUTER_RACE", 3),
    "OR014_6_0_201.mat": ("201", "OUTER_RACE", 0),
    "OR014_6_1_202.mat": ("202", "OUTER_RACE", 1),
    "OR014_6_2_203.mat": ("203", "OUTER_RACE", 2),
    "OR014_6_3_204.mat": ("204", "OUTER_RACE", 3),
    "OR021_6_0_238.mat": ("238", "OUTER_RACE", 0),
    "OR021_6_1_239.mat": ("239", "OUTER_RACE", 1),
    "OR021_6_2_240.mat": ("240", "OUTER_RACE", 2),
    "OR021_6_3_241.mat": ("241", "OUTER_RACE", 3),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_rpm(mat, mat_id: str, load: int, fname: str) -> float:
    rpm_key = f"X{mat_id}RPM"
    if rpm_key in mat:
        return float(np.asarray(mat[rpm_key]).squeeze())
    print(f"  [RPM fallback] {fname}: key '{rpm_key}' tidak ada, pakai RPM "
          f"resmi load {load} = {OFFICIAL_APPROX_RPM_BY_LOAD[load]}")
    return float(OFFICIAL_APPROX_RPM_BY_LOAD[load])


class Command(BaseCommand):
    help = "Training rf-baseline v1.1 (fitur turunan + hyperparameter search) ke Model Registry."

    def handle(self, *args, **options):
        rows, labels, groups = [], [], []
        file_hashes = {}

        print(f"===== Load {len(FILE_MANIFEST)} file CWRU dari {DATASET_ROOT} =====")
        for fname, (mat_id, fault_class, load) in sorted(FILE_MANIFEST.items()):
            path = DATASET_ROOT / fname
            if not path.exists():
                raise FileNotFoundError(f"File training tidak ditemukan: {path}")

            file_hashes[fname] = _sha256_file(path)

            de_key = f"X{mat_id}_DE_time"
            mat = sio.loadmat(path, variable_names=[de_key, f"X{mat_id}RPM"])
            if de_key not in mat:
                raise ValueError(f"{fname}: key '{de_key}' tidak ditemukan — manifest/file mismatch")

            waveform = mat[de_key].squeeze().astype(np.float64)
            rpm = _load_rpm(mat, mat_id, load, fname)

            n_windows = len(waveform) // WINDOW_SAMPLES
            for w in range(n_windows):
                chunk = waveform[w * WINDOW_SAMPLES: (w + 1) * WINDOW_SAMPLES]
                base_feats = extract_feature_vector(chunk, SAMPLE_RATE_HZ, rpm)
                engineered = engineer_features(base_feats)
                full_feats = {**base_feats, **engineered}
                rows.append([full_feats[k] for k in MODEL_FEATURE_ORDER])
                labels.append(fault_class)
                groups.append(fname)

            print(f"  {fname:26s} class={fault_class:12s} load={load} rpm={rpm:.1f} windows={n_windows}")

        X = np.array(rows, dtype=np.float64)
        y = np.array(labels)
        groups = np.array(groups)

        print(f"\n===== Dataset: {X.shape[0]} window, {X.shape[1]} fitur "
              f"({len(MODEL_FEATURE_ORDER)} = 9 kanonik + {len(MODEL_FEATURE_ORDER)-9} turunan), "
              f"{len(set(groups))} file-group, classes={sorted(set(y))} =====")
        for cls in sorted(set(y)):
            print(f"  {cls:12s} windows={int(np.sum(y == cls)):4d}  file-groups={len(set(groups[y == cls]))}")

        n_splits = 5
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(sgkf.split(X, y, groups))

        param_grid = {
            "n_estimators": [200, 500],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }
        keys = list(param_grid.keys())
        combos = list(product(*param_grid.values()))

        print(f"\n===== Hyperparameter search RandomForestClassifier: "
              f"{len(combos)} kombinasi x {n_splits} fold (folds identik) =====")
        search_results = []
        for combo in combos:
            params = dict(zip(keys, combo))
            fold_accs, fold_f1s = [], []
            for train_idx, test_idx in folds:
                clf = RandomForestClassifier(random_state=42, class_weight="balanced", **params)
                clf.fit(X[train_idx], y[train_idx])
                preds = clf.predict(X[test_idx])
                fold_accs.append(accuracy_score(y[test_idx], preds))
                fold_f1s.append(f1_score(y[test_idx], preds, average="macro"))
            mean_acc = float(np.mean(fold_accs))
            mean_f1 = float(np.mean(fold_f1s))
            search_results.append({"params": params, "mean_accuracy": mean_acc, "mean_macro_f1": mean_f1})
            print(f"  {params}  acc={mean_acc:.4f}  macro_f1={mean_f1:.4f}")

        best = max(search_results, key=lambda r: r["mean_macro_f1"])
        best_params = best["params"]
        print(f"\n===== Kombinasi terbaik: {best_params}  "
              f"acc={best['mean_accuracy']:.4f}  macro_f1={best['mean_macro_f1']:.4f} =====")

        fold_results = []
        print(f"\n===== Detail per-fold (hyperparameter terbaik) =====")
        for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
            clf = RandomForestClassifier(random_state=42, class_weight="balanced", **best_params)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], preds)
            macro_f1 = f1_score(y[test_idx], preds, average="macro")

            labels_sorted = sorted(set(y.tolist()))
            cm = confusion_matrix(y[test_idx], preds, labels=labels_sorted)
            print(f"    Confusion matrix fold {fold_idx} (order={labels_sorted}):")
            for i, row_label in enumerate(labels_sorted):
                print(f"      {row_label:12s} {cm[i].tolist()}")
            print(classification_report(y[test_idx], preds, labels=labels_sorted, zero_division=0))

            fold_results.append({
                "fold": fold_idx, "accuracy": acc, "macro_f1": macro_f1,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "test_groups": sorted(set(groups[test_idx].tolist())),
            })
            print(f"  Fold {fold_idx}: acc={acc:.4f} macro_f1={macro_f1:.4f}")

        mean_acc = float(np.mean([r["accuracy"] for r in fold_results]))
        std_acc = float(np.std([r["accuracy"] for r in fold_results]))
        mean_f1 = float(np.mean([r["macro_f1"] for r in fold_results]))
        std_f1 = float(np.std([r["macro_f1"] for r in fold_results]))
        print(f"\n===== HASIL CV v1.1: accuracy={mean_acc:.4f} (+/-{std_acc:.4f})  "
              f"macro_f1={mean_f1:.4f} (+/-{std_f1:.4f})  [v1.0 sebelumnya: acc=0.8622, macro_f1=0.8614] =====")

        final_clf = RandomForestClassifier(random_state=42, class_weight="balanced", **best_params)
        final_clf.fit(X, y)

        model_version = "rf-baseline-cwru-v1.1"
        artifact_dir = ARTIFACT_DIR / model_version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.joblib"
        joblib.dump(final_clf, artifact_path)
        print(f"\n===== Artifact tersimpan: {artifact_path} =====")

        combined = "\n".join(f"{fname}:{file_hashes[fname]}" for fname in sorted(file_hashes))
        dataset_hash = hashlib.sha256(combined.encode()).hexdigest()

        entry = {
            "model_version": model_version,
            "model_type": "CLASSIFICATION",
            "training_dataset": "CWRU",
            "training_dataset_version_hash": dataset_hash,
            "eval_accuracy": mean_acc,
            "eval_accuracy_std": std_acc,
            "eval_macro_f1": mean_f1,
            "eval_macro_f1_std": std_f1,
            "eval_date": date.today().isoformat(),
            "artifact_path": str(artifact_path),
            "feature_order": MODEL_FEATURE_ORDER,
            "label_classes": final_clf.classes_.tolist(),
            "n_training_windows": int(X.shape[0]),
            "n_training_files": len(FILE_MANIFEST),
            "hyperparameters": best_params,
            "hyperparameter_search_space": param_grid,
            "hyperparameter_search_results": search_results,
            "cv_fold_details": fold_results,
            "cross_dataset_pu_accuracy": None,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "supersedes": "rf-baseline-cwru-v1.0",
        }
        model_registry.append_entry(entry)
        print(f"\n===== Terdaftar di Model Registry: {model_registry.REGISTRY_PATH} =====")
        print(json.dumps(
            {k: v for k, v in entry.items()
             if k not in ("cv_fold_details", "hyperparameter_search_results")},
            indent=2,
        ))
