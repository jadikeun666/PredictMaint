"""
apps/ml_inference/pu_labels.py

Penurun label dataset Paderborn (PU) dari PDF dokumentasi per kode bearing
("Profile of rolling bearing damage", K001.pdf, KA01.pdf, ...). Pure python
(tanpa dependensi) supaya bisa dipakai alat host dan test.

KENAPA dari PDF, bukan dari nama kode: awalan KA/KI/KB menyesatkan. Terbukti dari
PDF: KI04 memuat kerusakan IR DAN OR (fatigue pitting + indentasi), KB23 memuat
IR, IR, OR. Label diturunkan dari baris `Component` tiap PDF:
    {}            + kode K00n      -> HEALTHY
    {OR}                           -> OUTER_RACE
    {IR}                           -> INNER_RACE
    {RE}                           -> BALL
    kombinasi lain / tidak dikenal -> MIXED  (dikeluarkan dari evaluasi 3-kelas)
Asal kerusakan: "artificial" (EDM, bearing tidak dioperasikan), "real" (uji umur /
fatigue), "healthy".
"""
import re

HEALTHY_CODE_RE = re.compile(r"^K00\d$")


def compact_lines(text: str):
    """Rapikan keluaran pdftotext: spasi berulang -> satu, buang baris kosong."""
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            out.append(ln)
    return out


def _first(lines, pattern):
    rx = re.compile(pattern)
    for ln in lines:
        m = rx.search(ln)
        if m:
            return m
    return None


def parse_pdf_text(text: str) -> dict:
    lines = compact_lines(text)
    full = "\n".join(lines)
    m = re.search(r"\(Bearing:\s*([A-Z0-9]+)\)", full)
    if not m:
        raise ValueError("kode bearing '(Bearing: XXXX)' tidak ditemukan pada teks PDF")
    code = m.group(1)

    def num(pattern, cast=float):
        mm = _first(lines, pattern)
        return cast(mm.group(1)) if mm else None

    geometry = {
        "pitch_diameter_mm": num(r"Pitch circle diameter\s+mm\s+([0-9.]+)"),
        "num_balls": num(r"Number of rolling elements\s+pc\.\s+(\d+)", int),
        "ball_diameter_mm": num(r"Rolling element diameter\s+mm\s+([0-9.]+)"),
        "contact_angle_deg": num(r"Nominal pressure angle\s+°\s+([0-9.]+)"),
    }
    mm = _first(lines, r"^Lifetime\s+h(?::min)?\s+(\S+)$")
    lifetime = mm.group(1) if mm else None
    mm = _first(lines, r"^Number of damages\s+(\d+)$")
    n_damages = int(mm.group(1)) if mm else None
    mm = _first(lines, r"^Mode\s+(.+)$")
    mode = mm.group(1) if mm else None
    mm = _first(lines, r"^Damage method\s+(.+)$")
    method = mm.group(1) if mm else None

    components = []
    for ln in lines:
        m2 = re.match(r"^Component((?: [A-Z]{2})+)$", ln)
        if m2:
            components = m2.group(1).split()
            break

    artificial = ("artificial damage, bearing was not operated" in full) or (mode is not None and "artificial" in mode.lower())
    real = ((method is not None and "lifetime test" in method.lower())
            or (lifetime is not None and re.match(r"^\d+:\d+$", lifetime) is not None))
    if HEALTHY_CODE_RE.match(code) and not components:
        origin = "healthy"
    elif artificial and not real:
        origin = "artificial"
    elif real and not artificial:
        origin = "real"
    else:
        origin = "unknown"

    entry = {
        "code": code, "geometry": geometry, "lifetime": lifetime, "n_damages": n_damages,
        "mode": mode, "damage_method": method, "components": components, "origin": origin,
    }
    entry["label"] = derive_label(entry)
    return entry


def derive_label(entry: dict) -> str:
    comps = set(entry["components"])
    if not comps:
        return "HEALTHY" if HEALTHY_CODE_RE.match(entry["code"]) else "UNKNOWN"
    if comps == {"OR"}:
        return "OUTER_RACE"
    if comps == {"IR"}:
        return "INNER_RACE"
    if comps == {"RE"}:
        return "BALL"
    return "MIXED"


def geometry_complete(entry: dict) -> bool:
    return all(v is not None for v in entry["geometry"].values())
