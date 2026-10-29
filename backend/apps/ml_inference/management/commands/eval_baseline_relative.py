"""
apps/ml_inference/management/commands/eval_baseline_relative.py

Sesi C, Giliran 1b — EKSPERIMEN offline: apakah fitur relatif-baseline
(rul_relative.py) memperbaiki LSTM-RUL? Protokol SAMA dengan evaluasi v1.0
(leave-one-trajectory-out, 3 seed, hyperparameter sama), dievaluasi pada jendela
dengan baseline lengkap (titik terakhir >= B); v1.0 dihitung ulang pada jendela
yang SAMA dari oof_predictions.npz. Tidak menyentuh jalur live/DB/registry.

ATURAN ADOPSI (ditetapkan sebelum melihat hasil): v1.1 diadopsi hanya bila
(a) ada kebijakan alert yang memenuhi kriteria (PREMATURE <= max_premature di
CRITICAL dan HIGH, rata-rata HIT >= min_hit) DAN (b) MAE rata-rata v1.1 < MAE
rata-rata v1.0 pada jendela yang sama. Selain itu: v1.0 tetap, tanpa alert RUL.
Estimasi durasi ~3 menit.
"""
import json
import os
import time

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from sklearn.linear_model import Ridge

from apps.ml_inference import ims_dataset as ims
from apps.ml_inference import rul_alert_policy as pol
from apps.ml_inference import rul_eval as ev
from apps.ml_inference import rul_relative as rr
from apps.ml_inference.lstm_rul import DEFAULT_HP, fit_lstm, predict_lstm
from apps.ml_inference.management.commands.register_lstm_rul import dataset_hash
from config import severity_thresholds as T

EVAL_DIR = ims.DEFAULT_CACHE_DIR.parent / "lstm-rul-eval"
METHODS = ("const", "ridge_rel", "lstm_v10", "lstm_v11")
KEYS = ("mae", "rmse", "mae_last10pct", "bias")


def _pct(x):
    return "   n/a" if x is None else f"{100 * x:5.0f}%"


class Command(BaseCommand):
    help = "Eksperimen offline: fitur relatif-baseline untuk LSTM-RUL."

    def add_arguments(self, parser):
        parser.add_argument("--cache-dir", default=str(ims.DEFAULT_CACHE_DIR))
        parser.add_argument("--eval-json", default=str(EVAL_DIR / "eval_results.json"))
        parser.add_argument("--out-dir", default=str(EVAL_DIR))
        parser.add_argument("--baseline-points", type=int, default=rr.BASELINE_POINTS)
        parser.add_argument("--factor", type=float, default=pol.DEFAULT_PREMATURE_FACTOR)
        parser.add_argument("--max-premature", type=float, default=0.25)
        parser.add_argument("--min-hit", type=float, default=0.5)

    def handle(self, *args, **o):
        B = o["baseline_points"]
        with open(o["eval_json"]) as f:
            ev_res = json.load(f)
        data = {}
        for t in ims.TRAJECTORIES:
            p = ims.cache_path(o["cache_dir"], t.traj_id)
            if not p.exists():
                raise CommandError(f"cache {p} tidak ada")
            data[t.traj_id] = ims.load_trajectory_cache(p)
        if dataset_hash(data) != ev_res["training_dataset_version_hash"]:
            raise CommandError("hash dataset != eval_results.json")
        hp, seeds = ev_res["config"]["hp"], ev_res["config"]["seeds"]
        if hp != DEFAULT_HP:
            raise CommandError("hyperparameter evaluasi != DEFAULT_HP")
        oof_path = os.path.join(o["out_dir"], "oof_predictions.npz")
        if not os.path.exists(oof_path):
            raise CommandError("oof_predictions.npz tidak ada — jalankan analyze_rul_alert_policy dulu")
        oof = np.load(oof_path)
        th = pol.Thresholds(T.RUL_CRITICAL_MAX_HOURS, T.RUL_HIGH_MAX_HOURS, T.RUL_WARNING_MAX_HOURS)

        rel = {t: rr.relative_features(d.primary, B) for t, d in data.items()}
        trajs = [(t.traj_id, t.set_no) for t in ims.TRAJECTORIES]
        rows, cases, cases_v10, npz = [], [], [], {}
        self.stdout.write(f"baseline = {B} snapshot pertama per bearing; jendela dievaluasi bila titik terakhir >= {B}")

        for fold, test_ids, train_ids in ev.build_folds(trajs, "loto"):
            sc = rr.fit_zscore([rel[t][B:] for t in train_ids])
            Xs, ys = [], []
            for t in train_ids:
                ws = ev.make_windows(rr.apply_zscore(sc, rel[t]), data[t].timestamps, data[t].rul_hours)
                m = rr.window_mask(ws.last_idx, B)
                Xs.append(ws.X[m]); ys.append(ws.y[m])
            Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
            ridge = Ridge(alpha=1.0).fit(Xtr.reshape(len(Xtr), -1), np.log1p(ytr))

            tid = test_ids[0]
            d = data[tid]
            L = float(d.rul_hours[0])
            ws = ev.make_windows(rr.apply_zscore(sc, rel[tid]), d.timestamps, d.rul_hours)
            m = rr.window_mask(ws.last_idx, B)
            Xte, yte = ws.X[m], ws.y[m]
            true_full = oof[f"{tid}__true"]
            m10 = (np.arange(len(true_full)) + 5) >= B
            if not np.allclose(true_full[m10], yte):
                raise CommandError(f"{tid}: label v1.0 dan v1.1 tidak sejajar")

            per = {k: [] for k in ("lstm_v11", "lstm_v10")}
            for sd in seeds:
                t0 = time.time()
                model, tstats, _ = fit_lstm(Xtr, ytr, sd, hp)
                p11 = predict_lstm(model, tstats, Xte)
                p10 = oof[f"{tid}__pred_seed{sd}"][m10]
                npz[f"{tid}__pred_seed{sd}"] = p11
                npz[f"{tid}__true"] = yte
                per["lstm_v11"].append(ev.regression_metrics(p11, yte, L))
                per["lstm_v10"].append(ev.regression_metrics(p10, yte, L))
                cases.append(pol.Case(tid, sd, yte, p11, L))
                cases_v10.append(pol.Case(tid, sd, yte, p10, L))
                self.stdout.write(f"  {tid} seed {sd}: MAE v1.1 {per['lstm_v11'][-1]['mae']:.1f} | "
                                  f"v1.0 {per['lstm_v10'][-1]['mae']:.1f} jam ({time.time()-t0:.0f}s)")
            res = {
                "const": ev.regression_metrics(np.full(len(yte), float(ytr.mean())), yte, L),
                "ridge_rel": ev.regression_metrics(np.clip(np.expm1(ridge.predict(Xte.reshape(len(Xte), -1))), 0, None), yte, L),
                "lstm_v10": {k: float(np.mean([r[k] for r in per["lstm_v10"]])) for k in KEYS},
                "lstm_v11": {k: float(np.mean([r[k] for r in per["lstm_v11"]])) for k in KEYS},
            }
            res["lstm_v11"]["mae_seed_std"] = float(np.std([r["mae"] for r in per["lstm_v11"]]))
            rows.append({"traj": tid, "traj_hours": L, "n_windows": int(len(yte)), "methods": res})

            P = np.stack([npz[f"{tid}__pred_seed{s}"] for s in seeds])
            first = yte > 0.5 * L
            last = yte <= 0.1 * L
            self.stdout.write(f"\n  == {tid}: umur {L:.0f} jam, {len(yte)} jendela | median prediksi v1.1: 50% awal {np.median(P[:, first]):.0f} jam "
                              f"(asli median {np.median(yte[first]):.0f}), 10% akhir {np.median(P[:, last]):.0f} jam")
            self.stdout.write(f"     {'metode':10s} {'MAE':>8s} {'RMSE':>8s} {'MAE10%':>8s} {'bias':>8s}")
            for mth in METHODS:
                r = res[mth]
                self.stdout.write(f"     {mth:10s} {r['mae']:8.1f} {r['rmse']:8.1f} {r['mae_last10pct']:8.1f} {r['bias']:8.1f}")

        # ---- agregat ----
        self.stdout.write("\n=== AGREGAT (4 lintasan, jendela baseline-lengkap; rata-rata ± std ANTAR-LINTASAN) ===")
        self.stdout.write(f"{'metode':10s} {'MAE(j)':>16s} {'RMSE(j)':>16s} {'MAE10%(j)':>16s}")
        agg = {}
        for mth in METHODS:
            agg[mth] = {k: (float(np.mean([r['methods'][mth][k] for r in rows])), float(np.std([r['methods'][mth][k] for r in rows])))
                        for k in ("mae", "rmse", "mae_last10pct")}
            a = agg[mth]
            self.stdout.write(f"{mth:10s} " + " ".join(f"{a[k][0]:9.1f}±{a[k][1]:6.1f}" for k in ("mae", "rmse", "mae_last10pct")))
        wins = sum(1 for r in rows if r["methods"]["lstm_v11"]["mae"] < min(r["methods"][m]["mae"] for m in ("const", "ridge_rel", "lstm_v10")))
        self.stdout.write(f"v1.1 terbaik (MAE) di {wins}/{len(rows)} lintasan; v1.1 lebih baik dari v1.0 di "
                          f"{sum(1 for r in rows if r['methods']['lstm_v11']['mae'] < r['methods']['lstm_v10']['mae'])}/{len(rows)}")

        # ---- simulasi kebijakan alert ----
        self.stdout.write(f"\n=== simulasi 24 kebijakan pada prediksi v1.1 (kriteria SAMA: PREMATURE <= {o['max_premature']:.0%}, rata-rata HIT >= {o['min_hit']:.0%}) ===")
        grid = [pol.evaluate_policy(cases, p, th, factor=o["factor"]) for p in pol.build_grid()]
        name, reason = pol.select_policy(grid, o["max_premature"], o["min_hit"])
        self.stdout.write(f"{'kebijakan':20s} | CRITICAL hit prem late miss (n) | HIGH hit prem late miss (n) | layak")
        for r in grid:
            c, h = r["levels"]["CRITICAL"], r["levels"]["HIGH"]
            ok = (c["n_eval"] and h["n_eval"] and c["premature_rate"] <= o["max_premature"] and h["premature_rate"] <= o["max_premature"]
                  and (c["hit_rate"] + h["hit_rate"]) / 2 >= o["min_hit"])
            self.stdout.write(f"{r['policy']:20s} | {_pct(c['hit_rate'])} {_pct(c['premature_rate'])} {_pct(c['late_rate'])} {_pct(c['miss_rate'])} ({c['n_eval']:2d}) | "
                              f"{_pct(h['hit_rate'])} {_pct(h['premature_rate'])} {_pct(h['late_rate'])} {_pct(h['miss_rate'])} ({h['n_eval']:2d}) |"
                              f"{'  <== TERPILIH' if r['policy'] == name else ('  ya' if ok else '')}")
        p0_v10 = pol.evaluate_policy(cases_v10, pol.BASELINE_POLICY, th, factor=o["factor"])
        c0, h0 = p0_v10["levels"]["CRITICAL"], p0_v10["levels"]["HIGH"]
        self.stdout.write(f"(pembanding v1.0, P0 pada jendela yang sama: CRITICAL hit {_pct(c0['hit_rate'])} prem {_pct(c0['premature_rate'])} | "
                          f"HIGH hit {_pct(h0['hit_rate'])} prem {_pct(h0['premature_rate'])})")

        better = agg["lstm_v11"]["mae"][0] < agg["lstm_v10"]["mae"][0]
        adopt = bool(name) and better
        self.stdout.write(f"\nKEPUTUSAN: {'ADOPSI v1.1 — kebijakan ' + name if adopt else 'TOLAK v1.1 (v1.0 tetap; jalur RUL tanpa alert)'} "
                          f"| kebijakan layak: {name or 'tidak ada'} ({reason}) | MAE v1.1 {agg['lstm_v11']['mae'][0]:.1f} vs v1.0 {agg['lstm_v10']['mae'][0]:.1f} jam "
                          f"({'lebih baik' if better else 'TIDAK lebih baik'})")

        os.makedirs(o["out_dir"], exist_ok=True)
        np.savez_compressed(os.path.join(o["out_dir"], "oof_predictions_rel.npz"), **npz)
        final = os.path.join(o["out_dir"], "relative_features_eval.json")
        with open(final + ".tmp", "w") as f:
            json.dump({"baseline_points": B, "rows": rows, "aggregate": agg, "grid": grid, "selected": name, "adopt": adopt}, f, indent=2, default=str)
        os.replace(final + ".tmp", final)
        self.stdout.write(f"hasil ditulis: {final}")
