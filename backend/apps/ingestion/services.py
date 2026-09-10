"""
Ingestion business logic — decode + persist a validated vibration window.

Dipanggil HANYA setelah validate_vibration_payload() (validation.py)
mengonfirmasi payload persis sesuai mqtt-architecture.md §Message Payload.
Modul ini tidak memvalidasi ulang — validation.py adalah satu-satunya
sumber kebenaran untuk itu (engineering-rules.md — no duplicate
validation logic).
"""
import base64
import os
import uuid

import numpy as np
from django.conf import settings
from django.utils.dateparse import parse_datetime

from apps.ingestion.models import VibrationWindow


def _waveform_path_for(sensor_id, captured_at_str):
    """
    {RAW_WAVEFORM_STORAGE_PATH}/{sensor_id}/{captured_at}_{rand8}.npy
    Dikelompokkan per sensor untuk penelusuran manual; suffix random 8-hex
    mencegah tabrakan nama file kalau dua window kebetulan share
    captured_at yang sama (belum ada idempotency check di giliran ini,
    lihat giliran 3).
    """
    safe_ts = captured_at_str.replace(":", "").replace("-", "")
    filename = f"{safe_ts}_{uuid.uuid4().hex[:8]}.npy"
    return os.path.join(settings.RAW_WAVEFORM_STORAGE_PATH, str(sensor_id), filename)


def ingest_vibration_window(sensor, payload):
    """
    Decode samples_b64 -> file .npy (dtype float32 tidak diubah, per
    signal-processing.md — resampling/precision-change hanya terjadi di
    titik pemrosesan lain, bukan di titik penyimpanan raw).

    Insert satu row vibration_windows immutable (engineering-rules.md
    §Data Integrity — tidak ada UPDATE).

    duration_s = num_samples / sample_rate_hz — field ini TIDAK ada di
    payload MQTT (mqtt-architecture.md §Message Payload), tapi computable
    dari dua field yang sudah tervalidasi, bukan fabrikasi.
    """
    num_samples = payload["num_samples"]
    sample_rate_hz = payload["sample_rate_hz"]
    captured_at = parse_datetime(payload["captured_at"])
    duration_s = round(num_samples / sample_rate_hz, 3)

    raw_bytes = base64.b64decode(payload["samples_b64"])
    samples = np.frombuffer(raw_bytes, dtype=np.float32)

    waveform_path = _waveform_path_for(sensor.id, payload["captured_at"])
    os.makedirs(os.path.dirname(waveform_path), exist_ok=True)
    np.save(waveform_path, samples)

    window = VibrationWindow.objects.create(
        sensor=sensor,
        captured_at=captured_at,
        duration_s=duration_s,
        num_samples=num_samples,
        waveform_path=waveform_path,
        source_dataset=payload["source_dataset"],
    )
    return window


def find_existing_window(sensor, captured_at):
    """
    Idempotency check (mqtt-architecture.md §QoS Policy — QoS 1 dapat
    menyebabkan redelivery/duplikat, ditangani di sini, bukan diabaikan).
    Application-level check (bukan DB unique constraint — tidak mengubah
    skema, engineering-rules.md/database.md tetap locked). Aman dari race
    condition karena on_message paho-mqtt berjalan single-threaded per
    proses ingestion-service.
    """
    return VibrationWindow.objects.filter(sensor=sensor, captured_at=captured_at).first()


def write_dead_letter(topic, raw_payload_bytes, errors):
    """
    Persist malformed/invalid payload ke dead-letter log (prd.md A1 AC
    kedua — tidak silently dropped). JSONL, satu baris per pesan gagal,
    ditulis ke volume raw-waveforms yang sama (tidak perlu volume baru).
    """
    import json as _json
    from datetime import datetime, timezone

    dead_letter_dir = os.path.join(settings.RAW_WAVEFORM_STORAGE_PATH, "_dead-letter")
    os.makedirs(dead_letter_dir, exist_ok=True)
    dead_letter_path = os.path.join(dead_letter_dir, "dead_letter.jsonl")

    try:
        raw_excerpt = raw_payload_bytes.decode("utf-8", errors="replace")[:2000]
    except Exception:
        raw_excerpt = repr(raw_payload_bytes)[:2000]

    entry = {
        "topic": topic,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "raw_payload_excerpt": raw_excerpt,
    }
    with open(dead_letter_path, "a") as f:
        f.write(_json.dumps(entry) + "\n")

    return dead_letter_path


def upsert_sensor_offline_alert(sensor):
    """
    SENSOR_OFFLINE alert dengan cooldown pattern (engineering-rules.md
    "Cooldown, not spam", database.md alerts.last_seen_at) — kalau alert
    unresolved untuk bearing ini sudah ada, update last_seen_at saja,
    jangan buat row baru.

    severity=HIGH adalah keputusan implementasi (scheduling.md §1 hanya
    mendefinisikan severity untuk RUL/klasifikasi, tidak untuk
    SENSOR_OFFLINE) — dicatat sebagai Addendum, bukan nilai dari spec.
    """
    from django.utils import timezone as _tz
    from apps.alerting.models import Alert

    now = _tz.now()
    existing = Alert.objects.filter(
        bearing=sensor.bearing,
        alert_type=Alert.AlertType.SENSOR_OFFLINE,
        resolved_at__isnull=True,
    ).first()
    if existing:
        existing.last_seen_at = now
        existing.save(update_fields=["last_seen_at"])
        return existing, False

    alert = Alert.objects.create(
        bearing=sensor.bearing,
        alert_type=Alert.AlertType.SENSOR_OFFLINE,
        severity=Alert.Severity.HIGH,
        triggered_at=now,
        last_seen_at=now,
    )
    return alert, True


def resolve_sensor_offline_alert(sensor):
    """
    Auto-resolve saat sensor kembali online (heartbeat baru masuk lagi).
    resolved_by=null (system-resolved) — pola sama seperti auto-resolve
    LOW_STOCK di inventory.md §4, diterapkan ke SENSOR_OFFLINE (deviasi
    kecil, inventory.md tidak eksplisit menyebut SENSOR_OFFLINE — dicatat
    di Addendum).
    """
    from django.utils import timezone as _tz
    from apps.alerting.models import Alert

    return Alert.objects.filter(
        bearing=sensor.bearing,
        alert_type=Alert.AlertType.SENSOR_OFFLINE,
        resolved_at__isnull=True,
    ).update(resolved_at=_tz.now())


def upsert_sensor_offline_alert(sensor):
    """
    SENSOR_OFFLINE alert dengan cooldown pattern (engineering-rules.md
    "Cooldown, not spam", database.md alerts.last_seen_at) — kalau alert
    unresolved untuk bearing ini sudah ada, update last_seen_at saja,
    jangan buat row baru.

    severity=HIGH adalah keputusan implementasi (scheduling.md §1 hanya
    mendefinisikan severity untuk RUL/klasifikasi, tidak untuk
    SENSOR_OFFLINE) — dicatat sebagai Addendum, bukan nilai dari spec.
    """
    from django.utils import timezone as _tz
    from apps.alerting.models import Alert

    now = _tz.now()
    existing = Alert.objects.filter(
        bearing=sensor.bearing,
        alert_type=Alert.AlertType.SENSOR_OFFLINE,
        resolved_at__isnull=True,
    ).first()
    if existing:
        existing.last_seen_at = now
        existing.save(update_fields=["last_seen_at"])
        return existing, False

    alert = Alert.objects.create(
        bearing=sensor.bearing,
        alert_type=Alert.AlertType.SENSOR_OFFLINE,
        severity=Alert.Severity.HIGH,
        triggered_at=now,
        last_seen_at=now,
    )
    return alert, True


def resolve_sensor_offline_alert(sensor):
    """
    Auto-resolve saat sensor kembali online (heartbeat baru masuk lagi).
    resolved_by=null (system-resolved) — pola sama seperti auto-resolve
    LOW_STOCK di inventory.md §4, diterapkan ke SENSOR_OFFLINE (deviasi
    kecil, inventory.md tidak eksplisit menyebut SENSOR_OFFLINE — dicatat
    di Addendum).
    """
    from django.utils import timezone as _tz
    from apps.alerting.models import Alert

    return Alert.objects.filter(
        bearing=sensor.bearing,
        alert_type=Alert.AlertType.SENSOR_OFFLINE,
        resolved_at__isnull=True,
    ).update(resolved_at=_tz.now())
