"""
apps/ml_inference/management/commands/build_pu_cache.py

Bangun cache evaluasi PU (fitur 9 + gelombang 12 kHz). OFFLINE; tidak menyentuh DB/InfluxDB.
  python manage.py build_pu_cache --limit-files 2 --out-dir /tmp/pu_smoke      # uji waktu
  python manage.py build_pu_cache                                              # penuh
"""
from collections import Counter

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from apps.ml_inference import ims_dataset as ims
from apps.ml_inference import pu_cache, pu_dataset as pu

DEFAULT_OUT = ims.DEFAULT_CACHE_DIR.parent / "pu_cache"


class Command(BaseCommand):
    help = "Ekstrak fitur + gelombang 12 kHz untuk evaluasi lintas-dataset PU."

    def add_arguments(self, parser):
        parser.add_argument("--pu-root", default=str(pu.DEFAULT_PU_ROOT))
        parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
        parser.add_argument("--workers", type=int, default=6)
        parser.add_argument("--limit-files", type=int, default=None, help="hanya N file pertama per kode (uji waktu)")
        parser.add_argument("--codes", nargs="+", default=None)

    def handle(self, *args, **o):
        try:
            labels = pu.load_labels()
        except FileNotFoundError:
            raise CommandError(f"{pu.LABELS_PATH} tidak ada — jalankan tools/build_pu_labels.py di host")
        codes = o["codes"] or pu.eval_codes(labels)
        c = Counter((labels[k]["label"], labels[k]["origin"]) for k in codes)
        self.stdout.write(f"{len(codes)} kode dievaluasi: {dict(c)}")
        st = pu_cache.build_cache(
            o["pu_root"], o["out_dir"], labels, codes, o["workers"], o["limit_files"],
            progress=lambda i, n, s: self.stdout.write(f"  {i}/{n} file ({s:.0f}s)"))
        self.stdout.write(f"jendela penuh per file (jumlah jendela: jumlah file): {st['windows_per_file']}")
        self.stdout.write(f"selesai: {st['n_files']} file -> {st['n_windows']} jendela dalam {st['seconds']:.0f}s "
                          f"({1000*st['seconds']/st['n_files']:.0f} ms/file, {o['workers']} proses)")
        if o["limit_files"]:
            full = sum(80 for _ in codes)
            self.stdout.write(f"estimasi penuh ({full} file): ~{st['seconds']/st['n_files']*full/60:.1f} menit")
        idx = np.load(f"{st['out_dir']}/index.npz")
        self.stdout.write(f"RPM di luar toleransi 3% terhadap kode kondisi: {st['n_rpm_insane']} jendela")
        for spd in ("09", "15"):
            m = idx["speed_code"] == spd
            if m.any():
                self.stdout.write(f"  N{spd}: RPM terukur rata-rata {idx['rpm'][m].mean():.1f} (min {idx['rpm'][m].min():.1f}, maks {idx['rpm'][m].max():.1f})")
        feats, lab = idx["features"], idx["label"]
        self.stdout.write(f"nilai non-finite pada fitur: {int((~np.isfinite(feats)).sum())}")
        for L in pu.EVAL_LABELS:
            m = lab == L
            if m.any():
                self.stdout.write(f"  {L:11s} {int(m.sum()):5d} jendela | rms median {np.median(feats[m, 0]):.3f} | kurtosis median {np.median(feats[m, 1]):.2f}")
