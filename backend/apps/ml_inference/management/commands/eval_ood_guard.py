"""
Sesi E, Giliran 1: evaluasi kandidat penjaga aplikabilitas model (OFFLINE, baca-saja).

Protokol terkunci di apps/ml_inference/ood_candidates.py (CONFIG + hash). Aturan:
- Kalibrasi/ambang/pemilihan HANYA dari 342 jendela CWRU (leave-one-group-out + nested tau).
- PU dan IMS hanya dilaporkan (tingkat penolakan; alarm RF sebelum vs sesudah), tidak
  menyetel apa pun. RF = rf-baseline-cwru-v1.1 (aturan alarm sekarang: kelas fault dengan
  keyakinan >= 0,6; CRITICAL bila >= 0,85).
- Tidak menulis ke DB/registry/InfluxDB. Keluaran: ml_artifacts/ood-guard-eval/.

Jalankan di container:
  docker compose run --rm -T --no-deps django-api python manage.py eval_ood_guard [--smoke]
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
from django.core.management.base import BaseCommand, CommandError

from apps.ml_inference import feature_extraction as fe
from apps.ml_inference import ood_candidates as oc
from apps.ml_inference.management.commands import train_rf_baseline as trb

LOCKED_HASH = "98a5d8697d80c9bd60636453d62935295a5b1c1ccf0a0801e1f35fd433fa2e97"
OUT_DIR = Path("/app/ml_artifacts/ood-guard-eval")
PU_INDEX = Path("/app/ml_artifacts/pu_cache/index.npz")
IMS_DIR = Path("/app/ml_artifacts/ims_features")
REGISTRY = Path("/app/config/model_registry.json")
IMS_TRAJ = ["IMS-S1-B3", "IMS-S1-B4", "IMS-S2-B1", "IMS-S3-B3"]
RF_VERSION = "rf-baseline-cwru-v1.1"


def build_cwru():
    rows, ys, gs = [], [], []
    for fname in sorted(trb.FILE_MANIFEST):
        mat_id, cls, load = trb.FILE_MANIFEST[fname]
        key = "X%s_DE_time" % mat_id
        mat = sio.loadmat(trb.DATASET_ROOT / fname, variable_names=[key, "X%sRPM" % mat_id])
        wf = mat[key].squeeze().astype(np.float64)
        rpm = trb._load_rpm(mat, mat_id, load, fname)
        gid = ("HEALTHY_L%d" % load) if cls == "HEALTHY" else fname.split("_")[0]
        for w in range(len(wf) // trb.WINDOW_SAMPLES):
            chunk = wf[w * trb.WINDOW_SAMPLES:(w + 1) * trb.WINDOW_SAMPLES]
            base = fe.extract_feature_vector(chunk, trb.SAMPLE_RATE_HZ, rpm)
            rows.append([base[k] for k in oc.FEATURES])
            ys.append(cls)
            gs.append(gid)
    return np.array(rows, dtype=np.float64), np.array(ys), np.array(gs)


class RF:
    def __init__(self):
        entries = [e for e in json.loads(REGISTRY.read_text()) if e["model_version"] == RF_VERSION]
        if not entries:
            raise CommandError("entri registry %s tidak ada" % RF_VERSION)
        reg = entries[-1]
        self.clf = joblib.load(reg["artifact_path"])
        self.order = reg["feature_order"]

    def predict(self, X9):
        mat = []
        for row in X9:
            base = dict(zip(oc.FEATURES, [float(v) for v in row]))
            full = {**base, **fe.engineer_features(base)}
            mat.append([full[k] for k in self.order])
        probs = self.clf.predict_proba(np.array(mat, dtype=np.float64))
        idx = probs.argmax(axis=1)
        pred = np.array([str(self.clf.classes_[i]) for i in idx])
        conf = probs[np.arange(len(idx)), idx]
        alarm = np.isin(pred, oc.CONFIG["rf_alarm_classes"]) & (conf >= oc.CONFIG["rf_alarm_min_conf"])
        crit = alarm & (conf >= oc.CONFIG["rf_critical_min_conf"])
        return pred, conf, alarm, crit


def pct(a, b):
    return (100.0 * a / b) if b else float("nan")


def summarize_subset(sel, reject, label, alarm, crit, pred):
    n = int(sel.sum())
    keep = sel & ~reject
    return {
        "n": n,
        "reject": int((sel & reject).sum()),
        "alarm_before": int((sel & alarm).sum()),
        "alarm_after": int((keep & alarm).sum()),
        "crit_before": int((sel & crit).sum()),
        "crit_after": int((keep & crit).sum()),
        "alarm_correct_before": int((sel & alarm & (pred == label)).sum()),
        "alarm_correct_after": int((keep & alarm & (pred == label)).sum()),
    }


class Command(BaseCommand):
    help = "Sesi E G1: evaluasi kandidat penjaga aplikabilitas (kalibrasi CWRU saja; PU/IMS hanya dilaporkan)."

    def add_arguments(self, parser):
        parser.add_argument("--smoke", action="store_true",
                            help="subsampel PU/IMS saja (protokol dan nested tetap penuh)")

    def handle(self, *args, **opts):
        smoke = opts["smoke"]
        h = oc.config_hash()
        if h != LOCKED_HASH:
            raise CommandError("CONFIG berbeda dari yang terkunci: %s" % h)
        if tuple(fe.FEATURE_ORDER) != tuple(oc.FEATURES):
            raise CommandError("FEATURE_ORDER tidak sama dengan oc.FEATURES")
        print("CONFIG_HASH cocok dengan kunci: %s" % h)
        t0 = time.time()

        X, y, g = build_cwru()
        if len(y) != 342:
            raise CommandError("jumlah jendela CWRU %d (harus 342)" % len(y))
        gids = sorted(set(g.tolist()))
        print("CWRU: %d jendela, %d grup: %s" % (len(y), len(gids), gids))
        print("  kelas: %s" % {c: int((y == c).sum()) for c in sorted(set(y.tolist()))})

        results = {}
        for kind in oc.CANDIDATES:
            ts = time.time()
            results[kind] = oc.evaluate_candidate(X, y, g, kind)
            print("  evaluasi %s selesai (%.1f dtk)" % (kind, time.time() - ts))

        selected, reason = oc.select_candidate(results)
        print("\n=== PEMILIHAN (aturan terkunci, hanya CWRU) ===")
        print("dipilih: %s -- %s" % (selected, reason))

        print("\n=== IN-DOMAIN: false-reject bersarang (makro antar-grup | maks) dan deteksi shift sintetis ===")
        for q in oc.CONFIG["quantiles"]:
            print("q=%.2f" % q)
            for kind in oc.CANDIDATES:
                if kind == "G1" and q != oc.CONFIG["quantiles"][0]:
                    continue
                pq = results[kind]["per_q"][q]
                worst = max(pq["frr"], key=pq["frr"].get)
                dm = pq["detection_macro"]
                print("  %s tau=%.4f FRR makro=%.4f maks=%.4f (%s) | deteksi amp2=%.3f amp5=%.3f impulsive=%.3f rata2=%.3f"
                      % (kind, results[kind]["final_tau"][q], pq["frr_macro"], pq["frr_max"], worst,
                         dm["amp2"], dm["amp5"], dm["impulsive"], pq["detection_mean"]))
        q0 = oc.CONFIG["selection_q"]
        print("\nFRR per grup pada q=%.2f (HEALTHY_L* = bocor sebagian):" % q0)
        print("  %-12s" % "grup" + "".join("%8s" % k for k in oc.CANDIDATES))
        for gid in gids:
            print("  %-12s" % gid + "".join("%8.3f" % results[k]["per_q"][q0]["frr"][gid] for k in oc.CANDIDATES))

        # ---------- PU (hanya dilaporkan) ----------
        rf = RF()
        d = np.load(PU_INDEX, allow_pickle=True)
        sane = d["rpm_sane"].astype(bool)
        if smoke:
            keep_idx = np.zeros(len(sane), dtype=bool)
            keep_idx[::25] = True
            sane = sane & keep_idx
        Xp = d["features"][sane]
        lab = d["label"][sane].astype(str)
        org = d["origin"][sane].astype(str)
        print("\n=== PU (dilaporkan, tidak menyetel apa pun): %d jendela rpm_sane%s ===" % (len(lab), " [SMOKE]" if smoke else ""))
        pred, conf, alarm, crit = rf.predict(Xp)
        hl = lab == "HEALTHY"
        print("RF sendiri: alarm (fault, conf>=0,6) pada sehat = %d/%d (%.1f%%); "
              "dari itu BALL/OUTER_RACE = %d (pembanding Sesi D: 907/1918 = 47,3%% untuk BALL/OUTER)"
              % ((hl & alarm).sum(), hl.sum(), pct((hl & alarm).sum(), hl.sum()),
                 (hl & alarm & np.isin(pred, ["BALL", "OUTER_RACE"])).sum()))
        combos = sorted(set(zip(lab.tolist(), org.tolist())))
        pu_report, pu_scores = {}, {}
        for kind in oc.CANDIDATES:
            gd = results[kind]["final_guard"]
            sc = gd.score(Xp)
            pu_scores[kind] = sc
            for q in ([None] if kind == "G1" else oc.CONFIG["quantiles"]):
                tau = 0.0 if kind == "G1" else results[kind]["final_tau"][q]
                rej = sc > tau
                tag = "%s@%s" % (kind, "tau0" if q is None else "q%.2f" % q)
                rep = {"ALL": summarize_subset(np.ones(len(lab), bool), rej, lab, alarm, crit, pred)}
                for (lb, og) in combos:
                    rep["%s/%s" % (lb, og)] = summarize_subset((lab == lb) & (org == og), rej, lab, alarm, crit, pred)
                pu_report[tag] = rep
                a = rep["ALL"]
                print("\n[%s] tau=%.4f  tolak %d/%d (%.1f%%)  | alarm RF %d -> %d, CRITICAL %d -> %d, alarm benar %d -> %d"
                      % (tag, tau, a["reject"], a["n"], pct(a["reject"], a["n"]), a["alarm_before"], a["alarm_after"],
                         a["crit_before"], a["crit_after"], a["alarm_correct_before"], a["alarm_correct_after"]))
                for key in sorted(k for k in rep if k != "ALL"):
                    r = rep[key]
                    print("    %-22s n=%5d tolak=%5.1f%% | alarm %5d -> %5d | CRITICAL %5d -> %5d | benar %5d -> %5d"
                          % (key, r["n"], pct(r["reject"], r["n"]), r["alarm_before"], r["alarm_after"],
                             r["crit_before"], r["crit_after"], r["alarm_correct_before"], r["alarm_correct_after"]))
                print("    pembanding penolakan acak pada laju tolak yang sama: alarm sehat diharapkan %.0f -> %.0f"
                      % ((hl & alarm).sum(), (hl & alarm).sum() * (1.0 - rej.mean())))

        # ---------- IMS (hanya dilaporkan) ----------
        print("\n=== IMS (dilaporkan): tingkat penolakan per fase umur; alarm RF sebelum -> sesudah ===")
        print("Catatan: fase awal diasumsikan sehat (tanpa label kelas valid); ini hanya tingkat penolakan.")
        ims_report = {}
        step = 5 if smoke else 1
        for tid in IMS_TRAJ:
            z = np.load(IMS_DIR / (tid + ".npz"), allow_pickle=True)
            meta = json.loads(str(z["meta_json"].item()))
            if tuple(meta["feature_order"]) != tuple(oc.FEATURES):
                raise CommandError("%s: feature_order cache IMS berbeda" % tid)
            Xi = z["features"][0][::step]
            rul = z["rul_hours"][::step]
            fin = np.isfinite(Xi).all(axis=1)
            Xi, rul = Xi[fin], rul[fin]
            elapsed = 1.0 - rul / z["rul_hours"].max()
            phases = {"awal(<%.0f%%)" % (100 * oc.CONFIG["ims_phases"]["early_elapsed_lt"]):
                      elapsed < oc.CONFIG["ims_phases"]["early_elapsed_lt"],
                      "akhir(>=%.0f%%)" % (100 * oc.CONFIG["ims_phases"]["late_elapsed_ge"]):
                      elapsed >= oc.CONFIG["ims_phases"]["late_elapsed_ge"]}
            ipred, iconf, ialarm, icrit = rf.predict(Xi)
            ims_report[tid] = {"n": int(len(Xi)), "dropped_nonfinite": int((~fin).sum())}
            print("\n%s: n=%d (non-finite dibuang: %d)" % (tid, len(Xi), int((~fin).sum())))
            for kind in oc.CANDIDATES:
                sc = results[kind]["final_guard"].score(Xi)
                for q in ([None] if kind == "G1" else oc.CONFIG["quantiles"]):
                    tau = 0.0 if kind == "G1" else results[kind]["final_tau"][q]
                    rej = sc > tau
                    tag = "%s@%s" % (kind, "tau0" if q is None else "q%.2f" % q)
                    parts = []
                    for pname, pm in phases.items():
                        n = int(pm.sum())
                        parts.append("%s n=%d tolak=%.1f%% alarm %d->%d" % (
                            pname, n, pct((pm & rej).sum(), n), (pm & ialarm).sum(), (pm & ~rej & ialarm).sum()))
                        ims_report[tid].setdefault(tag, {})[pname] = {
                            "n": n, "reject": int((pm & rej).sum()),
                            "alarm_before": int((pm & ialarm).sum()), "alarm_after": int((pm & ~rej & ialarm).sum())}
                    print("  %-10s %s" % (tag, " | ".join(parts)))

        print("\n=== G5 (allowlist provenance, opsi C) ===")
        print("Deterministik, bukan statistik: bearing_model terdaftar = 'SKF 6205-2RS JEM (drive end)' (CWRU) dan "
              "'Rexnord ZA-2115' (IMS x2). PU tidak terdaftar sebagai bearing di sistem. Tidak dievaluasi di sini.")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_smoke" if smoke else ""
        payload = {
            "config_hash": h, "smoke": smoke, "selected": selected, "selection_reason": reason,
            "cwru_groups": gids,
            "cwru": {k: {"final_tau": {str(q): v for q, v in results[k]["final_tau"].items()},
                         "per_q": {str(q): {"frr": results[k]["per_q"][q]["frr"],
                                            "frr_macro": results[k]["per_q"][q]["frr_macro"],
                                            "frr_max": results[k]["per_q"][q]["frr_max"],
                                            "detection": results[k]["per_q"][q]["detection"],
                                            "detection_macro": results[k]["per_q"][q]["detection_macro"],
                                            "detection_mean": results[k]["per_q"][q]["detection_mean"]}
                                   for q in oc.CONFIG["quantiles"]}}
                     for k in oc.CANDIDATES},
            "pu": pu_report, "ims": ims_report,
        }
        (OUT_DIR / ("eval_results%s.json" % suffix)).write_text(json.dumps(payload, indent=2, default=float))
        np.savez(OUT_DIR / ("pu_scores%s.npz" % suffix), label=lab, origin=org,
                 **{k: v for k, v in pu_scores.items()})
        print("\nSelesai (%.1f dtk). Keluaran: %s" % (time.time() - t0, OUT_DIR))
