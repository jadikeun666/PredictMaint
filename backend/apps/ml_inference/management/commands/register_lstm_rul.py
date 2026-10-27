"""
apps/ml_inference/management/commands/register_lstm_rul.py

Giliran 3 (Sesi B): latih model final lstm-rul pada SEMUA lintasan IMS
(hyperparameter + seed pertama sama dengan run evaluasi), simpan artefak,
dan daftarkan ke Model Registry (ml-pipeline.md §3) lewat model_registry.append_entry.

SEMUA angka evaluasi di entri registry dibaca dari eval_results.json hasil
`train_lstm_rul` (tidak ada yang diketik; engineering-rules.md). Command menolak
berjalan bila hash dataset / hyperparameter pada eval_results.json tidak cocok
dengan cache + DEFAULT_HP saat ini, atau bila model_version sudah terdaftar.

Catatan: angka evaluasi berasal dari model per-fold (leave-one-out, rata-rata
atas seed), BUKAN dari artefak final ini — dicatat eksplisit di entri.

Contoh:
  python manage.py register_lstm_rul --dry-run
  python manage.py register_lstm_rul
"""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from apps.ml_inference import ims_dataset as ims
from apps.ml_inference import model_registry
from apps.ml_inference import rul_eval as ev
from apps.ml_inference.feature_extraction import FEATURE_ORDER
from apps.ml_inference.lstm_rul import (
    DEFAULT_HP, fit_lstm, load_artifact, predict_lstm, save_artifact,
)

DEFAULT_VERSION = "lstm-rul-ims-v1.0"
EVAL_JSON = ims.DEFAULT_CACHE_DIR.parent / "lstm-rul-eval" / "eval_results.json"
BASELINES = ("const", "hi_extrap", "ridge")


def dataset_hash(data) -> str:
    """HARUS identik dengan perhitungan di train_lstm_rul (dicek lewat perbandingan hash)."""
    return hashlib.sha256(
        "\n".join(f"{k}:{data[k].meta['file_list_hash']}:{data[k].meta['cache_version']}"
                  for k in sorted(data)).encode()).hexdigest()


def build_registry_entry(ev_res, *, model_version, artifact_path, scaler, tstats, hp,
                         final_seed, n_train_windows, n_train_files, n_train_trajs,
                         trained_at, eval_date) -> dict:
    loto = ev_res["schemes"]["loto"]
    agg, rows = loto["aggregate"], loto["rows"]
    lm = agg["methods"]["lstm"]
    seeds = ev_res["config"]["seeds"]
    min_life = min(r["traj_hours"] for r in rows)
    rmse = float(lm["rmse"]["mean"])

    notes = [
        f"Dievaluasi leave-one-trajectory-out pada {agg['n_test_trajectories']} lintasan IMS, "
        f"{len(seeds)} seed; MAE {lm['mae']['mean']:.1f} jam, std ANTAR-LINTASAN {lm['mae']['std']:.1f} jam.",
        "Angka evaluasi berasal dari model per-fold (rata-rata atas seed), BUKAN dari artefak final ini "
        f"(artefak: dilatih pada semua lintasan, seed {final_seed}).",
    ]
    for name, sc in ev_res["schemes"].items():
        a = sc["aggregate"]
        notes.append(
            f"Skema {name}: LSTM mengalahkan semua baseline pada {a['lstm_beats_all_baselines_mae']}/"
            f"{a['n_test_trajectories']} lintasan (MAE); MAE LSTM {a['methods']['lstm']['mae']['mean']:.1f} vs "
            f"konstan {a['methods']['const']['mae']['mean']:.1f}, Ridge {a['methods']['ridge']['mae']['mean']:.1f} jam; "
            f"MAE 10% umur terakhir LSTM {a['methods']['lstm']['mae_last10pct']['mean']:.1f} vs "
            f"Ridge {a['methods']['ridge']['mae_last10pct']['mean']:.1f} jam.")
    notes.append("Bias per lintasan uji (jam, + = terlalu optimis): " + ", ".join(
        f"{r['test_traj']}={r['methods']['lstm']['bias']:+.1f}" for r in rows) + ".")
    notes.append(
        f"Pita kepercayaan = ± RMSE rata-rata LOTO ({rmse:.1f} jam): proksi sederhana, BUKAN interval Bayesian; "
        f"lebih lebar dari umur lintasan terpendek ({min_life:.1f} jam).")
    notes.append("Skor PHM adalah rata-rata exp() sehingga dapat meledak untuk prediksi sangat optimis; "
                 "baca bersama MAE dan bias.")
    notes.append("Jalur severity RUL (apps.alerting) belum divalidasi dengan prediksi RUL nyata.")

    return {
        "model_version": model_version,
        "model_type": "RUL",
        "training_dataset": "IMS",
        "training_dataset_version_hash": ev_res["training_dataset_version_hash"],
        "eval_accuracy": None,
        "eval_mae_hours": float(lm["mae"]["mean"]),
        "eval_mae_hours_std_across_trajectories": float(lm["mae"]["std"]),
        "eval_rmse_hours": rmse,
        "eval_mae_last10pct_hours": float(lm["mae_last10pct"]["mean"]),
        "eval_bias_hours_mean": float(lm["bias"]["mean"]),
        "eval_phm_score_mean": float(lm["phm"]["mean"]),
        "eval_protocol": "leave-one-trajectory-out (skema 'loto'), 3 seed; skema tambahan di eval_schemes",
        "eval_date": eval_date,
        "artifact_path": str(artifact_path),
        "trained_at": trained_at,
        "rul_confidence_band_hours": rmse,
        "rul_confidence_band_note": "± RMSE eval-set (rata-rata LOTO); proksi sederhana, bukan interval Bayesian",
        "feature_order": list(FEATURE_ORDER),
        "seq_len": ev.SEQ_LEN,
        "feature_transform": "signed_log1p lalu z-score (scaler dari semua lintasan latih)",
        "scaler_mean": [float(v) for v in scaler.mean],
        "scaler_std": [float(v) for v in scaler.std],
        "target_transform": "log1p(RUL jam), distandardisasi; inverse expm1, dipotong >= 0",
        "target_mu": float(tstats["mu"]),
        "target_sd": float(tstats["sd"]),
        "hyperparameters": dict(hp),
        "final_fit_seed": int(final_seed),
        "eval_seeds": list(seeds),
        "n_training_windows": int(n_train_windows),
        "n_training_files": int(n_train_files),
        "n_training_trajectories": int(n_train_trajs),
        "n_training_files_note": "dihitung per lintasan; file Set 1 dipakai dua kali (bearing 3 dan 4)",
        "cv_fold_details": [
            {"test_traj": r["test_traj"], "traj_hours": r["traj_hours"],
             "lstm": {k: r["methods"]["lstm"][k] for k in ("mae", "rmse", "phm", "mae_last10pct", "bias", "mae_seed_std")},
             "baselines_mae": {m: r["methods"][m]["mae"] for m in BASELINES}}
            for r in rows],
        "eval_schemes": {name: sc["aggregate"] for name, sc in ev_res["schemes"].items()},
        "evaluation_notes": notes,
        "supersedes": None,
    }


class Command(BaseCommand):
    help = "Latih model final lstm-rul pada semua lintasan IMS, simpan artefak, daftarkan ke Model Registry."

    def add_arguments(self, parser):
        parser.add_argument("--cache-dir", default=str(ims.DEFAULT_CACHE_DIR))
        parser.add_argument("--eval-json", default=str(EVAL_JSON))
        parser.add_argument("--model-version", default=DEFAULT_VERSION)
        parser.add_argument("--seed", type=int, default=None, help="default: seed pertama pada run evaluasi")
        parser.add_argument("--dry-run", action="store_true", help="latih + tampilkan entri, TANPA simpan/daftar")

    def handle(self, *args, **o):
        try:
            with open(o["eval_json"]) as f:
                ev_res = json.load(f)
        except FileNotFoundError:
            raise CommandError(f"{o['eval_json']} tidak ada — jalankan dulu: python manage.py train_lstm_rul")
        if "loto" not in ev_res.get("schemes", {}):
            raise CommandError("eval_results.json tidak memuat skema 'loto'")

        data = {}
        for t in ims.TRAJECTORIES:
            p = ims.cache_path(o["cache_dir"], t.traj_id)
            if not p.exists():
                raise CommandError(f"cache {p} tidak ada — jalankan build_ims_features")
            d = ims.load_trajectory_cache(p)
            if d.meta.get("cache_version") != ims.CACHE_VERSION:
                raise CommandError(f"cache {t.traj_id} versi lama")
            data[t.traj_id] = d
        if dataset_hash(data) != ev_res["training_dataset_version_hash"]:
            raise CommandError("hash dataset cache != hash pada eval_results.json — evaluasi basi, jalankan ulang train_lstm_rul")
        hp = ev_res["config"]["hp"]
        if hp != DEFAULT_HP:
            raise CommandError(f"hyperparameter evaluasi {hp} != DEFAULT_HP {DEFAULT_HP} — registrasi ditolak")
        if any(e.get("model_version") == o["model_version"] for e in model_registry.load_registry()):
            raise CommandError(f"model_version {o['model_version']} sudah ada di registry")

        seed = o["seed"] if o["seed"] is not None else ev_res["config"]["seeds"][0]
        ids = [t.traj_id for t in ims.TRAJECTORIES]
        sc = ev.fit_scaler([data[t].primary for t in ids])
        Xs, ys = [], []
        for t in ids:
            ws = ev.make_windows(ev.apply_scaler(sc, data[t].primary), data[t].timestamps, data[t].rul_hours)
            Xs.append(ws.X)
            ys.append(ws.y)
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)

        self.stdout.write(f"latih model final: {len(Xtr)} jendela dari {len(ids)} lintasan, seed {seed}, hp {hp}")
        model, tstats, loss = fit_lstm(Xtr, ytr, seed, hp)
        self.stdout.write(f"loss latih akhir (ternormalisasi): {loss:.4f}")

        now = datetime.now(timezone.utc)
        artifact_path = ims.DEFAULT_CACHE_DIR.parent / o["model_version"] / "model.pt"
        entry = build_registry_entry(
            ev_res, model_version=o["model_version"], artifact_path=artifact_path, scaler=sc,
            tstats=tstats, hp=hp, final_seed=seed, n_train_windows=len(Xtr),
            n_train_files=sum(data[t].meta["n_valid"] for t in ids), n_train_trajs=len(ids),
            trained_at=now.isoformat(timespec="seconds"), eval_date=now.date().isoformat())
        json.dumps(entry)                                      # pastikan serializable

        self.stdout.write("\nringkasan entri registry:")
        for k in ("model_version", "model_type", "training_dataset", "eval_mae_hours", "eval_mae_hours_std_across_trajectories",
                  "eval_rmse_hours", "eval_mae_last10pct_hours", "rul_confidence_band_hours", "eval_date", "artifact_path"):
            self.stdout.write(f"  {k}: {entry[k]}")
        self.stdout.write("catatan evaluasi tersimpan:")
        for n in entry["evaluation_notes"]:
            self.stdout.write(f"  - {n}")

        if o["dry_run"]:
            self.stdout.write(self.style.WARNING("\n--dry-run: artefak TIDAK disimpan, registry TIDAK diubah"))
            return

        save_artifact(artifact_path, model, tstats, sc.mean, sc.std, FEATURE_ORDER, ev.SEQ_LEN, hp,
                      meta={"model_version": o["model_version"], "final_fit_seed": seed,
                            "training_dataset_version_hash": entry["training_dataset_version_hash"],
                            "trained_at": entry["trained_at"]})
        m2, t2, mean2, std2, info = load_artifact(artifact_path)
        probe = Xtr[:: max(1, len(Xtr) // 500)]
        diff = float(np.max(np.abs(predict_lstm(model, tstats, probe) - predict_lstm(m2, t2, probe))))
        if diff > 1e-6 or not np.allclose(mean2, sc.mean) or not np.allclose(std2, sc.std):
            raise CommandError(f"artefak tidak identik setelah load (selisih prediksi {diff:.3g}) — registry TIDAK diubah")
        self.stdout.write(f"\nartefak disimpan + load ulang identik (selisih prediksi maks {diff:.2e}): {artifact_path}")

        model_registry.append_entry(entry)
        latest = model_registry.get_latest_entry("RUL")
        if latest is None or latest["model_version"] != o["model_version"]:
            raise CommandError("entri terdaftar tetapi get_latest_entry('RUL') tidak mengembalikannya")
        self.stdout.write(f"terdaftar di {model_registry.REGISTRY_PATH}; get_latest_entry('RUL') -> {latest['model_version']}")

        self.stdout.write("\nsanity IN-SAMPLE (model dilatih pada data ini; BUKAN bukti kualitas):")
        for t in ids:
            ws = ev.make_windows(ev.apply_scaler(sc, data[t].primary), data[t].timestamps, data[t].rul_hours)
            pr = predict_lstm(m2, t2, ws.X)
            for frac in (0.0, 0.5, 0.9, 1.0):
                i = min(len(ws.y) - 1, int(frac * (len(ws.y) - 1)))
                self.stdout.write(f"  {t} @{int(frac*100):3d}% umur: asli {ws.y[i]:8.1f} jam | prediksi {pr[i]:8.1f} jam")
