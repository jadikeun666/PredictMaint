"""
File-backed Model Registry (ml-pipeline.md §3 Model Registry Contract).

Storage decision (reported, tidak butuh approval — bukan perubahan skema
database.md): JSON file di backend/config/model_registry.json, dibaca
sekali oleh Celery worker saat startup dan di-append oleh training script
setelah eval selesai. Tidak ada dependency Django/ORM sehingga bisa
diimport baik dari management command (training) maupun dari Celery
worker (apps.ml_inference.tasks, inference serving) tanpa butuh konteks
Django request/DB penuh — rasional sama seperti
apps.signal_processing.bearing_frequencies (engineering-rules.md
§Code Organization).
"""
import json
import os
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "model_registry.json"

REQUIRED_FIELDS = [
    "model_version", "model_type", "training_dataset",
    "training_dataset_version_hash", "eval_date", "artifact_path",
]


def load_registry() -> list:
    """Semua entry model terdaftar, [] kalau file belum ada (training pertama kali)."""
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def append_entry(entry: dict) -> None:
    """
    Tambah satu entry model ke registry, atomik (tulis ke temp file lalu
    os.replace — supaya registry.json tidak setengah-tertulis kalau
    proses ke-kill di tengah jalan).
    """
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ValueError(
            f"Model registry entry kurang field wajib per "
            f"ml-pipeline.md §3 Model Registry Contract: {missing}"
        )

    registry = load_registry()
    if any(e.get("model_version") == entry["model_version"] for e in registry):
        raise ValueError(
            "model_version %s sudah ada di registry; naikkan versi atau gunakan "
            "update_entry_fields untuk mengisi field kosong (artefak tidak boleh ditimpa)"
            % entry["model_version"]
        )
    registry.append(entry)

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_path, REGISTRY_PATH)


def has_version(model_version: str) -> bool:
    """True bila model_version sudah terdaftar. Dipakai pemanggil untuk menolak SEBELUM menulis artefak."""
    return any(e.get("model_version") == model_version for e in load_registry())


def _is_deployable(entry: dict) -> bool:
    """Entri tanpa field `deployment` dianggap DEPLOYED (kompatibel dengan entri lama)."""
    return entry.get("deployment", "DEPLOYED") != "NOT_DEPLOYED"


def get_latest_entry(model_type: str):
    """
    Entry terbaru (by eval_date) untuk model_type tertentu
    ("CLASSIFICATION" atau "RUL") yang BOLEH disajikan: entri dengan
    deployment == "NOT_DEPLOYED" dilewati (terdaftar untuk jejak evaluasi,
    tidak dipilih worker inference). Dipakai worker inference saat startup
    (architecture.md section Model Loading) untuk memilih artifact_path.
    """
    candidates = [e for e in load_registry()
                  if e.get("model_type") == model_type and _is_deployable(e)]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["eval_date"])


def update_entry_fields(model_version: str, fields: dict) -> None:
    """
    Isi field pada SATU entri yang sudah ada. Hanya boleh mengisi field yang
    belum ada / masih None (registry append-only untuk nilai yang sudah
    tercatat); menolak versi yang tidak ada atau ambigu (duplikat).
    Atomik seperti append_entry.
    """
    registry = load_registry()
    matches = [e for e in registry if e.get("model_version") == model_version]
    if len(matches) != 1:
        raise ValueError("entri %s harus tepat satu, ditemukan %d" % (model_version, len(matches)))
    entry = matches[0]
    taken = [k for k in fields if entry.get(k) is not None]
    if taken:
        raise ValueError("field sudah terisi pada %s, tidak ditimpa: %s" % (model_version, taken))
    entry.update(fields)
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_path, REGISTRY_PATH)
