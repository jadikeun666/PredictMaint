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
    registry.append(entry)

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_path, REGISTRY_PATH)


def get_latest_entry(model_type: str):
    """
    Entry terbaru (by eval_date) untuk model_type tertentu
    ("CLASSIFICATION" atau "RUL"). Dipakai worker inference saat startup
    (architecture.md §Model Loading — load sekali, bukan per-task) untuk
    memilih artifact_path mana yang di-load.
    """
    candidates = [e for e in load_registry() if e.get("model_type") == model_type]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["eval_date"])
