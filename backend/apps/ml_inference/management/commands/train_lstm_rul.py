"""
apps/ml_inference/management/commands/train_lstm_rul.py

Giliran 2 (Sesi B): evaluasi OFFLINE lstm-rul dengan leave-one-trajectory-out
dan pembanding sederhana. TIDAK mendaftar ke Model Registry (Giliran 3) dan
tidak menyentuh DB/InfluxDB/worker. Membaca cache hasil `build_ims_features`.

Metode yang dibandingkan pada fold yang sama:
  const      : RUL konstan = rata-rata RUL lintasan latih
  hi_extrap  : ekstrapolasi linear HI (kausal) pada jendela 6 titik yang sama
  ridge      : regresi linear (Ridge) pada jendela 6 titik yang diratakan
  lstm       : LSTM kecil, rata-rata atas beberapa seed

Skema fold (rul_eval.SCHEMES): loto (spec), leave_set_out, loto_excl_set3.
Hasil lengkap ditulis ke backend/ml_artifacts/lstm-rul-eval/eval_results.json.
Semua angka berasal dari run ini (engineering-rules.md: tanpa angka ketikan).

Contoh:
  python manage.py train_lstm_rul --schemes loto --seeds 11 --epochs 3     # uji cepat
  python manage.py train_lstm_rul                                          # penuh
"""
import hashlib
import json
import os
import time

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from sklearn.linear_model import Ridge

from apps.ml_inference import ims_dataset as ims
from apps.ml_inference import rul_eval as ev
from apps.ml_inference.feature_extraction import FEATURE_ORDER
from apps.ml_inference.lstm_rul import DEFAULT_HP, fit_predict_lstm

DEFAULT_SEEDS = (11, 22, 33)
EVAL_DIR = ims.DEFAULT_CACHE_DIR.parent / "lstm-rul-eval"
METHODS = ("const", "hi_extrap", "ridge", "lstm")
METRIC_KEYS = ("mae", "rmse", "phm", "mae_last10pct", "bias")


def _fmt(v, w=9, p=2):
    return f"{v:{w}.{p}f}" if np.isfinite(v) else " " * (w - 3) + "nan"


class Command(BaseCommand):
    help = "Evaluasi LSTM-RUL (leave-one-trajectory-out) + baseline; offline, tanpa registrasi."

    def add_arguments(self, parser):
        parser.add_argument("--cache-dir", default=str(ims.DEFAULT_CACHE_DIR))
        parser.add_argument("--out-dir", default=str(EVAL_DIR))
        parser.add_argument("--schemes", nargs="+", default=list(ev.SCHEMES), choices=list(ev.SCHEMES))
        parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
        parser.add_argument("--epochs", type=int, default=None,
                            help="override epoch (hanya untuk uji cepat; hasil BUKAN untuk registry)")

    def handle(self, *args, **o):
        hp = dict(DEFAULT_HP)
        if o["epochs"] is not None:
            hp["epochs"] = o["epochs"]
            self.stdout.write(self.style.WARNING(f"--epochs={o['epochs']} aktif: hasil hanya uji cepat, BUKAN untuk registry"))
        seeds = o["seeds"]

        # --- muat cache ---
        data = {}
        for t in ims.TRAJECTORIES:
            p = ims.cache_path(o["cache_dir"], t.traj_id)
            if not p.exists():
                raise CommandError(f"cache {p} tidak ada — jalankan dulu: python manage.py build_ims_features")
            d = ims.load_trajectory_cache(p)
            if d.meta.get("cache_version") != ims.CACHE_VERSION:
                raise CommandError(f"cache {t.traj_id} versi lama — jalankan build_ims_features --force")
            data[t.traj_id] = d
        i_rms, i_kurt = FEATURE_ORDER.index("rms"), FEATURE_ORDER.index("kurtosis")
        ds_hash = hashlib.sha256(
            "\n".join(f"{k}:{data[k].meta['file_list_hash']}:{data[k].meta['cache_version']}"
                      for k in sorted(data)).encode()).hexdigest()

        self.stdout.write(f"hyperparameter LSTM (dikunci a priori): {hp}")
        self.stdout.write(f"seed: {seeds} | dataset hash: {ds_hash[:16]}…")
        self.stdout.write("lintasan:")
        for k, d in data.items():
            ws = ev.make_windows(d.primary, d.timestamps, d.rul_hours)
            self.stdout.write(f"  {k}: {len(d.timestamps)} titik, {len(ws.y)} jendela "
                              f"({int(ws.crosses_gap.sum())} melintasi celah >{ev.GAP_MINUTES:.0f} mnt), "
                              f"umur {d.rul_hours[0]:.1f} jam")

        trajs = [(t.traj_id, t.set_no) for t in ims.TRAJECTORIES]
        results = {"config": {"hp": hp, "seeds": seeds, "seq_len": ev.SEQ_LEN,
                              "gap_minutes": ev.GAP_MINUTES, "phm": [ev.PHM_A_EARLY, ev.PHM_A_LATE],
                              "feature_order": list(FEATURE_ORDER)},
                   "training_dataset_version_hash": ds_hash, "schemes": {}}
        t_start = time.time()

        for scheme in o["schemes"]:
            self.stdout.write(f"\n################ SKEMA: {scheme} ################")
            rows = []                                        # satu baris per (fold, lintasan uji)
            for fold_name, test_ids, train_ids in ev.build_folds(trajs, scheme):
                rows += self._run_fold(data, fold_name, test_ids, train_ids, hp, seeds, i_rms, i_kurt)
            results["schemes"][scheme] = {"rows": rows, "aggregate": self._aggregate(rows)}
            self.stdout.write(f"(waktu berjalan: {(time.time()-t_start)/60:.1f} menit)")

        os.makedirs(o["out_dir"], exist_ok=True)
        final = os.path.join(o["out_dir"], "eval_results.json")
        tmp = final + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, final)
        self.stdout.write(f"\nhasil lengkap ditulis: {final}")

    # ------------------------------------------------------------------
    def _run_fold(self, data, fold_name, test_ids, train_ids, hp, seeds, i_rms, i_kurt):
        raw_tr = [data[t].primary for t in train_ids]
        rul_tr = [data[t].rul_hours for t in train_ids]
        sc = ev.fit_scaler(raw_tr)
        Xs, ys = [], []
        for t in train_ids:
            ws = ev.make_windows(ev.apply_scaler(sc, data[t].primary), data[t].timestamps, data[t].rul_hours)
            Xs.append(ws.X)
            ys.append(ws.y)
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        hip = ev.fit_hi_params(raw_tr, rul_tr, i_rms, i_kurt)
        ridge = Ridge(alpha=1.0).fit(Xtr.reshape(len(Xtr), -1), np.log1p(ytr))

        te = {}
        for t in test_ids:
            d = data[t]
            te[t] = ev.make_windows(ev.apply_scaler(sc, d.primary), d.timestamps, d.rul_hours)
        Xte = np.concatenate([te[t].X for t in test_ids])
        sizes = [len(te[t].y) for t in test_ids]
        splits = np.cumsum(sizes)[:-1]

        self.stdout.write(f"\n--- fold uji={test_ids} | latih={train_ids} | {len(Xtr)} jendela latih")
        lstm_preds = []                                       # per seed: list per lintasan uji
        for sd in seeds:
            t0 = time.time()
            pred, loss = fit_predict_lstm(Xtr, ytr, Xte, sd, hp)
            lstm_preds.append(np.split(pred, splits))
            self.stdout.write(f"  seed {sd}: loss latih akhir (ternormalisasi) {loss:.4f} ({time.time()-t0:.0f}s)")

        rows = []
        for k, t in enumerate(test_ids):
            d, ws = data[t], te[t]
            L = float(d.rul_hours[0])
            per_method = {}
            per_method["const"] = ev.regression_metrics(ev.baseline_constant(rul_tr, len(ws.y)), ws.y, L)
            hp_pred, fb = ev.hi_extrapolation_predict(d.primary, d.timestamps, ws.last_idx, hip, i_rms, i_kurt)
            per_method["hi_extrap"] = ev.regression_metrics(hp_pred, ws.y, L)
            per_method["hi_extrap"]["fallback_frac"] = float(fb.mean())
            rp = np.clip(np.expm1(ridge.predict(ws.X.reshape(len(ws.X), -1))), 0.0, None)
            per_method["ridge"] = ev.regression_metrics(rp, ws.y, L)
            seed_metrics = [ev.regression_metrics(lstm_preds[s][k], ws.y, L) for s in range(len(seeds))]
            lm = {key: float(np.nanmean([m[key] for m in seed_metrics])) for key in METRIC_KEYS}
            lm["mae_seed_std"] = float(np.std([m["mae"] for m in seed_metrics]))
            lm["n"] = len(ws.y)
            per_method["lstm"] = lm

            self.stdout.write(f"  uji {t}: umur {L:.1f} jam, {len(ws.y)} jendela")
            self.stdout.write(f"    {'metode':10s} {'MAE(j)':>9s} {'RMSE(j)':>9s} {'PHM':>9s} {'MAE10%':>9s} {'bias(j)':>9s}")
            for m in METHODS:
                r = per_method[m]
                extra = f"  (fallback {100*r['fallback_frac']:.0f}%)" if m == "hi_extrap" else (
                    f"  (std MAE antar-seed {r['mae_seed_std']:.1f})" if m == "lstm" else "")
                self.stdout.write(f"    {m:10s} {_fmt(r['mae'])} {_fmt(r['rmse'])} {_fmt(r['phm'])} "
                                  f"{_fmt(r['mae_last10pct'])} {_fmt(r['bias'])}{extra}")
            rows.append({"fold": fold_name, "test_traj": t, "traj_hours": L, "train_trajs": train_ids,
                         "n_train_windows": int(len(Xtr)), "methods": per_method,
                         "lstm_seed_metrics": seed_metrics})
        return rows

    # ------------------------------------------------------------------
    def _aggregate(self, rows):
        agg = {"n_test_trajectories": len(rows), "methods": {}}
        for m in METHODS:
            agg["methods"][m] = {}
            for key in METRIC_KEYS:
                v = np.array([r["methods"][m][key] for r in rows], dtype=np.float64)
                agg["methods"][m][key] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
        wins = sum(1 for r in rows
                   if r["methods"]["lstm"]["mae"] < min(r["methods"][m]["mae"] for m in ("const", "hi_extrap", "ridge")))
        wins_phm = sum(1 for r in rows
                       if r["methods"]["lstm"]["phm"] < min(r["methods"][m]["phm"] for m in ("const", "hi_extrap", "ridge")))
        agg["lstm_beats_all_baselines_mae"] = wins
        agg["lstm_beats_all_baselines_phm"] = wins_phm

        self.stdout.write(f"\n=== AGREGAT ({len(rows)} lintasan uji; rata-rata ± std ANTAR-LINTASAN) ===")
        self.stdout.write(f"{'metode':10s} {'MAE(j)':>16s} {'RMSE(j)':>16s} {'PHM':>16s} {'MAE10%(j)':>16s}")
        for m in METHODS:
            a = agg["methods"][m]
            cells = [f"{a[k]['mean']:9.2f}±{a[k]['std']:6.2f}" for k in ("mae", "rmse", "phm", "mae_last10pct")]
            self.stdout.write(f"{m:10s} " + " ".join(f"{c:>16s}" for c in cells))
        self.stdout.write(f"LSTM mengalahkan SEMUA baseline pada {wins}/{len(rows)} lintasan (MAE), "
                          f"{wins_phm}/{len(rows)} (skor PHM)")
        return agg
