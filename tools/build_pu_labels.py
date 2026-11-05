#!/usr/bin/env python3
"""
tools/build_pu_labels.py — alat HOST (butuh `pdftotext`, poppler-utils).

Membaca PDF dokumentasi per kode bearing PU, menurunkan label/geometri/asal
kerusakan lewat apps/ml_inference/pu_labels.py, memvalidasi terhadap fakta yang
sudah terbukti dari PDF, lalu menulis
backend/apps/ml_inference/data/pu_bearing_labels.json (dibaca container; tidak
butuh pdftotext di container).

Pemakaian:
  python3 tools/build_pu_labels.py --pu-root "/mnt/c/.../MFPT Bearing Dataset"
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pu_labels", ROOT / "backend/apps/ml_inference/pu_labels.py")
pu_labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pu_labels)

# Fakta yang sudah terbukti dari teks PDF (dicetak pada recon Sesi D):
KNOWN = {
    "K001": dict(label="HEALTHY", origin="healthy", components=[]),
    "KA01": dict(label="OUTER_RACE", origin="artificial", components=["OR"]),
    "KA04": dict(label="OUTER_RACE", origin="real", components=["OR"], lifetime="12:01"),
    "KI01": dict(label="INNER_RACE", origin="artificial", components=["IR"]),
    "KI04": dict(label="MIXED", origin="real", components=["IR", "OR"]),
    "KB23": dict(label="MIXED", origin="real", components=["IR", "IR", "OR"], lifetime="21:01"),
}


def pdf_text(path: Path) -> str:
    return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pu-root", required=True)
    ap.add_argument("--out", default=str(ROOT / "backend/apps/ml_inference/data/pu_bearing_labels.json"))
    args = ap.parse_args()
    if shutil.which("pdftotext") is None:
        sys.exit("pdftotext tidak ditemukan (pasang poppler-utils)")

    root = Path(args.pu_root)
    pdfs = sorted(p for p in root.glob("K*/K*.pdf") if not p.name.startswith("measuring_log"))
    if not pdfs:
        sys.exit(f"tidak ada PDF K*/K*.pdf di {root}")

    entries, problems = {}, []
    for p in pdfs:
        try:
            e = pu_labels.parse_pdf_text(pdf_text(p))
        except Exception as ex:                                   # noqa: BLE001
            problems.append(f"{p.name}: gagal parse ({ex})")
            continue
        e["source_pdf"] = f"{p.parent.name}/{p.name}"
        e["source_pdf_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        if e["code"] != p.stem:
            problems.append(f"{p.name}: kode di dalam PDF ({e['code']}) != nama file")
        if not pu_labels.geometry_complete(e):
            problems.append(f"{p.name}: geometri tidak lengkap {e['geometry']}")
        if e["label"] in ("UNKNOWN",) or e["origin"] == "unknown":
            problems.append(f"{p.name}: label/asal tidak terturunkan (label={e['label']}, origin={e['origin']})")
        if e["label"] != "HEALTHY" and e["n_damages"] is not None and len(e["components"]) != e["n_damages"]:
            problems.append(f"{p.name}: jumlah komponen ({len(e['components'])}) != Number of damages ({e['n_damages']})")
        entries[e["code"]] = e

    for code, want in KNOWN.items():
        got = entries.get(code)
        if got is None:
            problems.append(f"{code}: fakta terbukti tidak bisa dicek (kode tidak ada)")
            continue
        for k, v in want.items():
            if got.get(k) != v:
                problems.append(f"{code}: {k} = {got.get(k)!r}, diharapkan {v!r} (fakta dari PDF)")
    if "K001" in entries:
        g = entries["K001"]["geometry"]
        if (g["ball_diameter_mm"], g["pitch_diameter_mm"], g["num_balls"], g["contact_angle_deg"]) != (6.75, 29.05, 8, 0.0):
            problems.append(f"K001: geometri {g} != (6.75, 29.05, 8, 0) yang terbukti dari PDF")

    print(f"{'kode':6s} {'label':11s} {'asal':10s} {'komponen':12s} {'umur':7s} {'d/D/n/phi'}")
    for code in sorted(entries):
        e = entries[code]; g = e["geometry"]
        print(f"{code:6s} {e['label']:11s} {e['origin']:10s} {','.join(e['components']) or '-':12s} {str(e['lifetime'] or '-'):7s} "
              f"{g['ball_diameter_mm']}/{g['pitch_diameter_mm']}/{g['num_balls']}/{g['contact_angle_deg']}")
    print("\nrekap (label, asal):", dict(Counter((e["label"], e["origin"]) for e in entries.values())))
    print("geometri unik:", sorted({tuple(e["geometry"].values()) for e in entries.values()}, key=str))

    if problems:
        print("\nMASALAH (hasil TIDAK ditulis):")
        for pr in problems:
            print("  -", pr)
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "PDF dokumentasi PU per kode bearing (Paderborn, KAt-DataCenter)",
        "generated_by": "tools/build_pu_labels.py",
        "label_rules": "komponen {OR}=OUTER_RACE, {IR}=INNER_RACE, {}=HEALTHY (K00n), lainnya=MIXED",
        "codes": entries,
    }, indent=1, sort_keys=True))
    print(f"\nditulis: {out} ({len(entries)} kode)")


if __name__ == "__main__":
    main()
