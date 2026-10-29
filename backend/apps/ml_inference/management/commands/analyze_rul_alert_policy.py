"""
apps/ml_inference/management/commands/analyze_rul_alert_policy.py

Sesi C, Giliran 1 — analisis OFFLINE. Tidak menyentuh jalur live, DB, InfluxDB,
atau registry. Tiga bagian:
  1. Latih ulang LSTM leave-one-trajectory-out dengan hyperparameter + seed yang
     SAMA dengan run evaluasi, lalu cek MAE-nya cocok dengan eval_results.json
     (bukti reproduksi) dan simpan prediksi out-of-fold.
  2. Simulasikan 24 kebijakan alert pada prediksi itu (rul_alert_policy.py) dan
     pilih dengan aturan yang ditetapkan sebelum melihat hasil.
  3. Diagnostik: apa yang dilakukan rf-baseline (CWRU-only) pada fitur IMS?

Hasil: backend/ml_artifacts/lstm-rul-eval/{oof_predictions.npz, alert_policy_analysis.json}
Estimasi durasi: ~3-4 menit (12 pelatihan kecil + simulasi).
"""
import json
import os
import time

import joblib
import numpy as np
from django.core.management.base import BaseCommand, CommandError

from apps.ml_inference import ims_dataset as ims
from apps.ml_inference import model_registry
from apps.ml_inference import rul_alert_policy as pol
from apps.ml_inference import rul_eval as ev
from apps.ml_inference.feature_extraction import FEATURE_ORDER, engineer_features
from apps.ml_inference.lstm_rul import DEFAULT_HP, fit_lstm, predict_lstm
from apps.ml_inference.management.commands.register_lstm_rul import dataset_hash
from config import severity_thresholds as T

EVAL_DIR = ims.DEFAULT_CACHE_DIR.parent / "lstm-rul-eval"


def _pct(x):
    return "   n/a" if x is None else f"{100 * x:5.0f}%"


class Command(BaseCommand):
    help = "Analisis kebijakan alert berbasis RUL dari prediksi out-of-fold (offline)."

    def add_arguments(self, parser):
        parser.add_argument("--cache-dir", default=str(ims.DEFAULT_CACHE_DIR))
        parser.add_argument("--eval-json", default=str(EVAL_DIR / "eval_results.json"))
        parser.add_argument("--out-dir", default=str(EVAL_DIR))
        parser.add_argument("--factor", type=float, default=pol.DEFAULT_PREMATURE_FACTOR)
        parser.add_argument("--max-premature", type=float, default=0.25)
        parser.add_argument("--min-hit", type=float, default=0.5)

    # ------------------------------------------------------------------
    def handle(self, *args, **o):
        try:
            with open(o["eval_json"]) as f:
                ev_res = json.load(f)
        except FileNotFoundError:
            raise CommandError(f"{o['eval_json']} tidak ada — jalankan dulu train_lstm_rul")
        data = {}
        for t in ims.TRAJECTORIES:
            p = ims.cache_path(o["cache_dir"], t.traj_id)
            if not p.exists():
                raise CommandError(f"cache {p} tidak ada — jalankan build_ims_features")
            data[t.traj_id] = ims.load_trajectory_cache(p)
        if dataset_hash(data) != ev_res["training_dataset_version_hash"]:
            raise CommandError("hash dataset != eval_results.json — evaluasi basi")
        hp = ev_res["config"]["hp"]
        if hp != DEFAULT_HP:
            raise CommandError("hyperparameter evaluasi != DEFAULT_HP")
        seeds = ev_res["config"]["seeds"]
        th = pol.Thresholds(T.RUL_CRITICAL_MAX_HOURS, T.RUL_HIGH_MAX_HOURS, T.RUL_WARNING_MAX_HOURS)

        cases, npz = self._oof(data, ev_res, hp, seeds)
        os.makedirs(o["out_dir"], exist_ok=True)
        np.savez_compressed(os.path.join(o["out_dir"], "oof_predictions.npz"), **npz)

        results = self._simulate(cases, th, o)
        diag = self._classifier_diagnostic(data)

        final = os.path.join(o["out_dir"], "alert_policy_analysis.json")
        with open(final + ".tmp", "w") as f:
            json.dump({"thresholds": vars(th) if hasattr(th, "__dict__") else str(th),
                       "factor": o["factor"], "max_premature": o["max_premature"], "min_hit": o["min_hit"],
                       "lead_min_hours": {pol.LEVEL_NAMES[k]: v for k, v in pol.DEFAULT_LEAD_MIN_HOURS.items()},
                       "seeds": seeds, "results": results["grid"], "selected": results["selected"],
                       "selection_reason": results["reason"], "classifier_on_ims": diag}, f, indent=2, default=str)
        os.replace(final + ".tmp", final)
        self.stdout.write(f"\nhasil ditulis: {final}")

    # ------------------------------------------------------------------
    def _oof(self, data, ev_res, hp, seeds):
        self.stdout.write("=== 1. prediksi out-of-fold (LOTO) + cek reproduksi ===")
        trajs = [(t.traj_id, t.set_no) for t in ims.TRAJECTORIES]
        cases, npz = [], {}
        eval_rows = {r["test_traj"]: r for r in ev_res["schemes"]["loto"]["rows"]}
        for fold, test_ids, train_ids in ev.build_folds(trajs, "loto"):
            sc = ev.fit_scaler([data[t].primary for t in train_ids])
            Xs, ys = [], []
            for t in train_ids:
                ws = ev.make_windows(ev.apply_scaler(sc, data[t].primary), data[t].timestamps, data[t].rul_hours)
                Xs.append(ws.X)
                ys.append(ws.y)
            Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
            tid = test_ids[0]
            ws_te = ev.make_windows(ev.apply_scaler(sc, data[tid].primary), data[tid].timestamps, data[tid].rul_hours)
            L = float(data[tid].rul_hours[0])
            npz[f"{tid}__true"] = ws_te.y
            maes = []
            for sd in seeds:
                t0 = time.time()
                model, tstats, _ = fit_lstm(Xtr, ytr, sd, hp)
                pred = predict_lstm(model, tstats, ws_te.X)
                maes.append(float(np.mean(np.abs(pred - ws_te.y))))
                npz[f"{tid}__pred_seed{sd}"] = pred
                cases.append(pol.Case(tid, sd, ws_te.y, pred, L))
                self.stdout.write(f"  {tid} seed {sd}: MAE {maes[-1]:.2f} jam ({time.time()-t0:.0f}s)")
            ref = eval_rows[tid]["methods"]["lstm"]["mae"]
            diff = abs(float(np.mean(maes)) - ref)
            self.stdout.write(f"  >> {tid}: MAE rata-rata seed {np.mean(maes):.3f} vs eval_results.json {ref:.3f} "
                              f"(selisih {diff:.2e}) {'REPRODUKSI OK' if diff < 1e-3 else '<<< TIDAK REPRODUKSI'}")
        return cases, npz

    # ------------------------------------------------------------------
    def _simulate(self, cases, th, o):
        self.stdout.write(f"\n=== 2. simulasi 24 kebijakan (ambang {th.critical:.0f}/{th.high:.0f}/{th.warning:.0f} jam; "
                          f"PREMATURE bila RUL asli > {o['factor']}x ambang; lead-min "
                          f"CRITICAL {pol.DEFAULT_LEAD_MIN_HOURS[3]:.0f} jam, HIGH {pol.DEFAULT_LEAD_MIN_HOURS[2]:.0f} jam) ===")
        n_traj = len({c.traj_id for c in cases})
        self.stdout.write(f"kasus = {n_traj} lintasan x {len({c.seed for c in cases})} seed = {len(cases)}; "
                          f"IMS-S2-B1 (umur 163,8 jam < lead-min HIGH) = NA di tingkat HIGH")
        grid_res = [pol.evaluate_policy(cases, p, th, factor=o["factor"]) for p in pol.build_grid()]
        name, reason = pol.select_policy(grid_res, o["max_premature"], o["min_hit"])

        self.stdout.write(f"\n{'kebijakan':20s} | CRITICAL hit prem late miss (n) | HIGH hit prem late miss (n) | layak")
        for r in grid_res:
            c, h = r["levels"]["CRITICAL"], r["levels"]["HIGH"]
            ok = (c["n_eval"] and h["n_eval"] and c["premature_rate"] <= o["max_premature"]
                  and h["premature_rate"] <= o["max_premature"]
                  and (c["hit_rate"] + h["hit_rate"]) / 2 >= o["min_hit"])
            mark = "  <== TERPILIH" if r["policy"] == name else ("  ya" if ok else "")
            self.stdout.write(
                f"{r['policy']:20s} | {_pct(c['hit_rate'])} {_pct(c['premature_rate'])} {_pct(c['late_rate'])} "
                f"{_pct(c['miss_rate'])} ({c['n_eval']:2d}) | {_pct(h['hit_rate'])} {_pct(h['premature_rate'])} "
                f"{_pct(h['late_rate'])} {_pct(h['miss_rate'])} ({h['n_eval']:2d}) |{mark}")
        self.stdout.write(f"\nKEPUTUSAN: {name if name else 'TIDAK ADA kebijakan yang layak'} — {reason}")

        show = [("P0 (perilaku saat ini)", pol.BASELINE_POLICY.name)]
        if name:
            show.append(("terpilih", name))
        by_name = {r["policy"]: r for r in grid_res}
        for label, nm in show:
            self.stdout.write(f"\n--- rincian per kasus: {label} = {nm}")
            for row in by_name[nm]["per_case"]:
                def fmt(lv):
                    d = row[lv]
                    tr = "-" if d["true_rul_at_alert"] is None else f"{d['true_rul_at_alert']:.0f}"
                    return f"{d['outcome']:9s} (RUL asli saat alert {tr:>5s} jam)"
                self.stdout.write(f"  {row['traj_id']:10s} seed {row['seed']:3d} | CRITICAL {fmt('CRITICAL')} | HIGH {fmt('HIGH')}")
        return {"grid": grid_res, "selected": name, "reason": reason}

    # ------------------------------------------------------------------
    def _classifier_diagnostic(self, data):
        self.stdout.write("\n=== 3. diagnostik: rf-baseline (CWRU-only) pada fitur IMS ===")
        entry = model_registry.get_latest_entry("CLASSIFICATION")
        if entry is None:
            self.stdout.write("  (tidak ada model CLASSIFICATION terdaftar)")
            return None
        clf = joblib.load(entry["artifact_path"])
        order = entry["feature_order"]
        self.stdout.write(f"  model: {entry['model_version']} (training_dataset={entry['training_dataset']}); "
                          f"ambang UNKNOWN 0,6; CRITICAL >= 0,85")
        out = {}
        for tid, d in data.items():
            rows = []
            for r in d.primary:
                base = dict(zip(FEATURE_ORDER, r))
                full = {**base, **engineer_features(base)}
                rows.append([full[k] for k in order])
            proba = clf.predict_proba(np.array(rows))
            idx = proba.argmax(axis=1)
            conf = proba.max(axis=1)
            cls = np.array([str(clf.classes_[i]) for i in idx])
            cls = np.where(conf < 0.6, "UNKNOWN", cls)
            n = len(cls)
            alert = np.isin(cls, ["INNER_RACE", "OUTER_RACE", "BALL"])
            crit = alert & (conf >= 0.85)
            late = np.arange(n) >= int(0.9 * n)
            dist = {c: int((cls == c).sum()) for c in ("HEALTHY", "INNER_RACE", "OUTER_RACE", "BALL", "UNKNOWN")}
            out[tid] = {"n": n, "dist": dist, "alert_share": float(alert.mean()), "critical_share": float(crit.mean()),
                        "alert_share_last10pct": float(alert[late].mean()), "alert_share_first90pct": float(alert[~late].mean())}
            self.stdout.write(f"  {tid}: {dist} | jendela memicu alert {100*alert.mean():.0f}% "
                              f"(CRITICAL {100*crit.mean():.0f}%) | 90% awal {100*alert[~late].mean():.0f}% vs 10% akhir {100*alert[late].mean():.0f}%")
        return out
