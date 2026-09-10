"""
Celery task(s) for the `signal_processing` queue.

GILIRAN 5 SCOPE: full 10-step signal-processing.md §Pipeline complete.
Step 9 writes InfluxDB derived_features; step 10 enqueues ml_inference
(task not implemented yet — BATASAN 2, this only enqueues by string
task name, same queue-boundary pattern ingestion-service already uses
for THIS task).

Task name/kwargs FIXED by the live ingestion-service:
    apps.signal_processing.tasks.process_vibration_window(window_id=<uuid str>)

This task, in turn, FIXES the name/kwargs for the NEXT session's
ml_inference task (mirrors signal-processing.md step 10's literal
wording "enqueue ... with the bearing_id and the new feature point"):
    apps.ml_inference.tasks.run_inference(
        bearing_id=<uuid str>, window_id=<uuid str>,
        captured_at=<isoformat str>, features=<dict>)
"""
import logging

import numpy as np
from celery import shared_task

from apps.ingestion.models import VibrationWindow
from apps.signal_processing.dsp_features import (
    remove_dc,
    apply_hann_window,
    compute_time_domain_features,
    compute_fft,
    compute_dominant_frequency,
)
from apps.signal_processing.envelope_features import (
    compute_envelope_spectrum,
    lookup_amplitude_near,
)
from apps.signal_processing.bearing_frequencies import compute_fault_frequencies
from apps.signal_processing.influx_client import (
    get_latest_rpm_at_or_before,
    write_derived_features,
)
from apps.signal_processing.alerts import (
    upsert_sensor_config_missing_alert,
    resolve_sensor_config_missing_alert,
)
from config.celery import app as celery_app

logger = logging.getLogger("signal_processing")


def _bearing_geometry_complete(bearing) -> bool:
    return None not in (
        bearing.num_balls,
        bearing.ball_diameter_mm,
        bearing.pitch_diameter_mm,
        bearing.contact_angle_deg,
    )


@shared_task(name="apps.signal_processing.tasks.process_vibration_window")
def process_vibration_window(window_id: str):
    window = VibrationWindow.objects.select_related("sensor__bearing__machine").get(id=window_id)
    sensor = window.sensor
    bearing = sensor.bearing
    machine = bearing.machine

    waveform = np.load(window.waveform_path).astype(np.float64, copy=False)
    sample_rate_hz = sensor.sample_rate_hz
    dc_removed = remove_dc(waveform)

    time_domain_features = compute_time_domain_features(dc_removed)

    hann_windowed = apply_hann_window(dc_removed)
    freqs_hz, magnitude = compute_fft(hann_windowed, sample_rate_hz)

    dominant_freq_hz = compute_dominant_frequency(freqs_hz, magnitude)

    bpfo_amp = bpfi_amp = bsf_amp = ftf_amp = None
    geometry_complete = _bearing_geometry_complete(bearing)
    rpm = get_latest_rpm_at_or_before(str(sensor.id), window.captured_at)
    fr_hz = (rpm / 60.0) if rpm is not None else None

    if geometry_complete and fr_hz is not None:
        fault_freqs = compute_fault_frequencies(
            fr_hz=fr_hz,
            num_balls=bearing.num_balls,
            ball_diameter_mm=bearing.ball_diameter_mm,
            pitch_diameter_mm=bearing.pitch_diameter_mm,
            contact_angle_deg=bearing.contact_angle_deg,
        )
        env_freqs, env_spectrum = compute_envelope_spectrum(dc_removed, sample_rate_hz)
        bpfo_amp = lookup_amplitude_near(env_freqs, env_spectrum, fault_freqs.bpfo)
        bpfi_amp = lookup_amplitude_near(env_freqs, env_spectrum, fault_freqs.bpfi)
        bsf_amp = lookup_amplitude_near(env_freqs, env_spectrum, fault_freqs.bsf)
        ftf_amp = lookup_amplitude_near(env_freqs, env_spectrum, fault_freqs.ftf)
        resolve_sensor_config_missing_alert(bearing)
    else:
        reasons = []
        if not geometry_complete:
            reasons.append("bearing geometry incomplete")
        if fr_hz is None:
            reasons.append("no RPM heartbeat recorded yet (sensor_status)")
        logger.warning(
            "[SENSOR_CONFIG_MISSING] window_id=%s bearing_id=%s reason=%s",
            window_id, bearing.id, "; ".join(reasons),
        )
        upsert_sensor_config_missing_alert(bearing)

    features = {
        **time_domain_features,
        "dominant_freq_hz": dominant_freq_hz,
        "bpfo_amp": bpfo_amp,
        "bpfi_amp": bpfi_amp,
        "bsf_amp": bsf_amp,
        "ftf_amp": ftf_amp,
    }

    write_derived_features(
        machine_id=machine.id,
        bearing_id=bearing.id,
        sensor_id=sensor.id,
        captured_at=window.captured_at,
        features=features,
    )

    try:
        celery_app.send_task(
            "apps.ml_inference.tasks.run_inference",
            kwargs={
                "bearing_id": str(bearing.id),
                "window_id": str(window_id),
                "captured_at": window.captured_at.isoformat(),
                "features": features,
            },
            queue="ml_inference",
        )
        logger.info(f"[ENQUEUE] window_id={window_id} -> queue=ml_inference")
    except Exception:
        logger.exception(f"[ENQUEUE ERROR] window_id={window_id} — gagal enqueue ke ml_inference")

    logger.info(
        "[GILIRAN5] window_id=%s sensor_id=%s bearing=%s rms=%.6f "
        "dominant_freq_hz=%.3f bpfo_amp=%s bpfi_amp=%s bsf_amp=%s ftf_amp=%s "
        "fr_hz=%s — written to InfluxDB derived_features, enqueued ml_inference",
        window_id, sensor.id, bearing.position_label,
        time_domain_features["rms"], dominant_freq_hz,
        bpfo_amp, bpfi_amp, bsf_amp, ftf_amp, fr_hz,
    )

    return {"window_id": str(window_id), **features, "fr_hz": fr_hz}
