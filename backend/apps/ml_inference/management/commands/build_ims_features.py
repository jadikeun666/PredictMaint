"""
apps/ml_inference/management/commands/build_ims_features.py

Ekstrak fitur 9-kanonik seluruh snapshot lintasan IMS run-to-failure, hitung
label RUL + Health Indicator, simpan cache .npz di backend/ml_artifacts/ims_features/.
OFFLINE saja. Tidak menyentuh DB/InfluxDB/registry.

Snapshot "sinyal mati" (akuisisi berhenti) dibuang dari input; waktu kegagalan
tetap file terakhir di disk. Lintasan dengan use_all_files=True (Set 3) memakai
semua file di disk, bukan hanya sampai batas tanggal Readme.

Contoh:
  python manage.py build_ims_features --only IMS-S2-B1 --limit 40   # uji waktu, tidak menyimpan
  python manage.py build_ims_features --workers 10 --force          # penuh, simpan cache
"""
import time

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from scipy.stats import spearmanr

from apps.ml_inference import ims_dataset as ims
from apps.ml_inference.feature_extraction import FEATURE_ORDER


class Command(BaseCommand):
    help = "Bangun cache fitur + label RUL/HI untuk lintasan IMS (lstm-rul, offline)."

    def add_arguments(self, parser):
        parser.add_argument("--ims-root", default=str(ims.DEFAULT_IMS_ROOT))
        parser.add_argument("--out-dir", default=str(ims.DEFAULT_CACHE_DIR))
        parser.add_argument("--only", action="append", default=None,
                            help="traj_id tertentu (boleh diulang), default semua")
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--limit", type=int, default=None,
                            help="hanya N snapshot pertama, UJI WAKTU saja (cache tidak ditulis)")
        parser.add_argument("--force", action="store_true", help="bangun ulang walau cache valid")

    def handle(self, *args, **o):
        ids = o["only"] or [t.traj_id for t in ims.TRAJECTORIES]
        trajs = []
        for tid in ids:
            try:
                trajs.append(ims.get_trajectory(tid))
            except KeyError as e:
                raise CommandError(str(e))

        for traj in trajs:
            self.stdout.write(f"\n===== {traj.traj_id} (Set {traj.set_no}, bearing {traj.bearing_no}, {traj.failure_mode}) =====")
            listing = ims.discover_snapshots(traj, o["ims_root"])
            self._report_listing(traj, listing)

            cpath = ims.cache_path(o["out_dir"], traj.traj_id)
            data = None
            if cpath.exists() and not o["force"] and not o["limit"]:
                cached = ims.load_trajectory_cache(cpath)
                if (cached.meta.get("file_list_hash") == ims.file_list_hash(listing.entries)
                        and cached.meta.get("cache_version") == ims.CACHE_VERSION):
                    self.stdout.write(f"cache valid ({cpath}), dipakai ulang — gunakan --force untuk bangun ulang")
                    data = cached
                else:
                    self.stdout.write("cache ada tapi tidak cocok (daftar file / versi berbeda) -> dibangun ulang")

            if data is None:
                t0 = time.time()
                data = ims.build_trajectory(
                    traj, listing, workers=o["workers"], limit=o["limit"],
                    progress=lambda i, n: self.stdout.write(f"  {i}/{n} snapshot ({time.time()-t0:.0f}s)"),
                )
                elapsed = time.time() - t0
                n = data.meta["n_files"]
                self.stdout.write(f"selesai {n} snapshot x {len(traj.columns)} kanal dalam {elapsed:.1f}s "
                                  f"({elapsed/n*1000:.0f} ms/snapshot, {o['workers']} worker)")
                if o["limit"]:
                    full = len(listing.entries)
                    self.stdout.write(f"  estimasi penuh lintasan ini: ~{elapsed/n*full/60:.1f} menit")
                    self.stdout.write("  (--limit aktif: cache TIDAK ditulis)")
                else:
                    self.stdout.write(f"cache ditulis: {ims.save_trajectory_cache(data, o['out_dir'])}")

            self._report_data(data)

    # ---- laporan verifikasi ----
    def _report_listing(self, traj, lst):
        e = lst.entries
        self.stdout.write(f"file di folder       : {lst.n_in_folder}")
        self.stdout.write(f"non-snapshot diabaikan: {lst.n_ignored_non_snapshot}")
        if traj.use_all_files:
            self.stdout.write(f"setelah batas Readme  : {lst.n_after_readme_end} (DIPAKAI — use_all_files, deviasi disengaja)")
        else:
            self.stdout.write(f"setelah batas Readme  : {lst.n_after_readme_end} (dikecualikan)")
        same = len(e) == traj.readme_n_files
        tag = "OK" if same else ("<<< BEDA dari Readme (disengaja: use_all_files)" if traj.use_all_files else "<<< SELISIH vs Readme")
        self.stdout.write(f"dipakai              : {len(e)} (Readme: {traj.readme_n_files}) {tag}")
        if e:
            self.stdout.write(f"file pertama         : {e[0].name} (Readme: {traj.readme_start}) "
                              f"{'OK' if e[0].name == traj.readme_start else '<<< BEDA'}")
            last_ok = e[-1].name == traj.readme_end
            last_tag = "OK" if last_ok else ("beda dari Readme (disengaja: use_all_files)" if traj.use_all_files else "<<< BEDA")
            self.stdout.write(f"file terakhir        : {e[-1].name} (Readme: {traj.readme_end}) {last_tag}")

    def _report_data(self, d):
        ts = d.timestamps
        m = d.meta
        dt = np.diff(ts) / 60.0
        self.stdout.write(f"snapshot valid       : {m['n_valid']} dari {m['n_files']} | sinyal mati dibuang: {m['n_dead_excluded']} {m['dead_files']}")
        self.stdout.write(f"akhir rekaman (gagal): {m['failure_file']} | snapshot valid terakhir: {m['last_valid_file']} (RUL {d.rul_hours[-1]:.2f} jam)")
        self.stdout.write(f"durasi lintasan      : {(ts[-1]-ts[0])/3600:.1f} jam valid | RUL awal {d.rul_hours[0]:.1f} jam")
        if dt.size:
            self.stdout.write(f"interval (menit)     : median {np.median(dt):.1f} | min {dt.min():.1f} | max {dt.max():.1f} | celah >15 menit: {int((dt > 15).sum())}")
        rho = spearmanr(ts, d.hi).correlation if len(ts) > 2 else float("nan")
        self.stdout.write(f"HI                   : awal {d.hi[0]:.3f} | akhir {d.hi[-1]:.3f} | Spearman(HI, waktu) = {rho:.3f}")
        bad = int((~np.isfinite(d.features)).sum())
        self.stdout.write(f"nilai non-finite     : {bad}")
        self.stdout.write("fitur kanal primer (min / median / max):")
        p = d.primary
        for j, name in enumerate(FEATURE_ORDER):
            col = p[:, j]
            self.stdout.write(f"  {name:17s} {col.min():12.4f} {np.median(col):12.4f} {col.max():12.4f}")
