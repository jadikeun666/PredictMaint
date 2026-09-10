"""
Payload validation for MQTT ingestion, PERSIS terhadap skema di
mqtt-architecture.md §Message Payload — `.../vibration` dan `.../status`.

Pure functions, tanpa efek samping (tidak menyentuh DB) — dipanggil dari
management command `run_ingestion`. Setiap kegagalan validasi menghasilkan
daftar error message untuk dead-letter log (giliran 3), BUKAN silently
dropped (prd.md A1 AC kedua).
"""
from datetime import datetime
import base64

VALID_AXES = {"RADIAL", "AXIAL"}
VALID_SOURCE_DATASETS = {"CWRU", "PU", "IMS", "SIMULATED_SYNTHETIC", "HARDWARE"}
VALID_STATUS_VALUES = {"ONLINE", "OFFLINE"}


def _parse_iso8601(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_vibration_payload(payload):
    """Returns (is_valid, errors). Required: captured_at, sample_rate_hz,
    num_samples, axis, source_dataset, samples_b64 (mqtt-architecture.md)."""
    errors = []
    if not isinstance(payload, dict):
        return False, ["payload is not a JSON object"]

    required = ["captured_at", "sample_rate_hz", "num_samples", "axis", "source_dataset", "samples_b64"]
    missing = [f for f in required if f not in payload]
    if missing:
        errors.append(f"missing required field(s): {', '.join(missing)}")

    captured_at = payload.get("captured_at")
    if "captured_at" in payload and _parse_iso8601(captured_at) is None:
        errors.append(f"captured_at is not a valid ISO8601 timestamp: {captured_at!r}")

    sample_rate_hz = payload.get("sample_rate_hz")
    if "sample_rate_hz" in payload and not (
        isinstance(sample_rate_hz, int) and not isinstance(sample_rate_hz, bool) and sample_rate_hz > 0
    ):
        errors.append(f"sample_rate_hz must be a positive integer, got {sample_rate_hz!r}")

    num_samples = payload.get("num_samples")
    if "num_samples" in payload and not (
        isinstance(num_samples, int) and not isinstance(num_samples, bool) and num_samples > 0
    ):
        errors.append(f"num_samples must be a positive integer, got {num_samples!r}")

    axis = payload.get("axis")
    if "axis" in payload and axis not in VALID_AXES:
        errors.append(f"axis must be one of {sorted(VALID_AXES)}, got {axis!r}")

    source_dataset = payload.get("source_dataset")
    if "source_dataset" in payload and source_dataset not in VALID_SOURCE_DATASETS:
        errors.append(f"source_dataset must be one of {sorted(VALID_SOURCE_DATASETS)}, got {source_dataset!r}")

    samples_b64 = payload.get("samples_b64")
    if "samples_b64" in payload:
        if not isinstance(samples_b64, str) or not samples_b64:
            errors.append("samples_b64 must be a non-empty string")
        else:
            try:
                decoded = base64.b64decode(samples_b64, validate=True)
            except Exception as exc:
                errors.append(f"samples_b64 is not valid base64: {exc}")
            else:
                # mqtt-architecture.md: "<base64-encoded float32 array>" -> 4 bytes/sample.
                if isinstance(num_samples, int) and not isinstance(num_samples, bool) and num_samples > 0:
                    expected_bytes = num_samples * 4
                    if len(decoded) != expected_bytes:
                        errors.append(
                            f"decoded samples_b64 length ({len(decoded)} bytes) != "
                            f"num_samples*4 ({expected_bytes} bytes, float32) — possible truncation/mismatch"
                        )

    return (len(errors) == 0), errors


def validate_status_payload(payload):
    """Returns (is_valid, errors). Required: status, rpm, last_seen
    (mqtt-architecture.md §Message Payload — .../status)."""
    errors = []
    if not isinstance(payload, dict):
        return False, ["payload is not a JSON object"]

    required = ["status", "rpm", "last_seen"]
    missing = [f for f in required if f not in payload]
    if missing:
        errors.append(f"missing required field(s): {', '.join(missing)}")

    status = payload.get("status")
    if "status" in payload and status not in VALID_STATUS_VALUES:
        errors.append(f"status must be one of {sorted(VALID_STATUS_VALUES)}, got {status!r}")

    rpm = payload.get("rpm")
    if "rpm" in payload and not (isinstance(rpm, (int, float)) and not isinstance(rpm, bool) and rpm > 0):
        errors.append(f"rpm must be a positive number, got {rpm!r}")

    last_seen = payload.get("last_seen")
    if "last_seen" in payload and _parse_iso8601(last_seen) is None:
        errors.append(f"last_seen is not a valid ISO8601 timestamp: {last_seen!r}")

    return (len(errors) == 0), errors
