"""
PredictMaint — MQTT Dataset-Replay Simulator (core publisher)

Standalone script, NOT part of the Django app. Reads a CWRU .mat file's
drive-end accelerometer channel, chunks it into 1.0s windows at native
sample rate, and publishes each window to the broker per the exact
payload schema in mqtt-architecture.md §Message Payload.

Also publishes a retained heartbeat to .../status every 30s (wall-clock,
independent of --speed-multiplier), with an LWT for fast offline
detection on unexpected disconnect (mqtt-architecture.md §Sensor Offline
Detection).

Handles both SIGINT (Ctrl+C) and SIGTERM (`kill`, container/systemd
stop) as graceful-shutdown triggers, so an explicit OFFLINE status is
always published on intentional stop — LWT is reserved for genuinely
unexpected disconnects (crash, network loss, power loss), not for any
form of deliberate process termination.
"""
import argparse
import base64
import json
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io
import paho.mqtt.client as mqtt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HEARTBEAT_INTERVAL_S = 30  # fixed wall-clock, per mqtt-architecture.md §Sensor Offline Detection


class GracefulExit(Exception):
    """Raised by the SIGTERM handler so `kill <pid>` unwinds through the
    same try/except/finally path as Ctrl+C (KeyboardInterrupt/SIGINT)."""
    pass


def _handle_sigterm(signum, frame):
    raise GracefulExit()


def load_env(env_path: Path) -> dict:
    """Minimal stdlib .env parser (no python-dotenv dependency)."""
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


def get_setting(env_file: dict, key: str) -> str:
    value = os.environ.get(key) or env_file.get(key)
    if not value:
        raise RuntimeError(f"Missing required setting: {key} (checked process env and .env)")
    return value


def read_de_channel(mat_path: Path, mat_id: str) -> np.ndarray:
    """Read the drive-end accelerometer channel per signal-processing.md
    §Dataset-Specific Ingestion Notes (drive-end selected by convention)."""
    data = scipy.io.loadmat(str(mat_path))
    key = f"X{mat_id}_DE_time"
    if key not in data:
        available = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"Channel key '{key}' not found in {mat_path.name}. Available keys: {available}")
    return data[key].flatten().astype(np.float64)


def chunk_plan(samples: np.ndarray, window_samples: int):
    n_complete = len(samples) // window_samples
    dropped = len(samples) - (n_complete * window_samples)
    return n_complete, dropped


def build_vibration_payload(window_f64: np.ndarray, sample_rate_hz: int, axis: str, source_dataset: str) -> dict:
    window_f32 = window_f64.astype(np.float32)
    samples_b64 = base64.b64encode(window_f32.tobytes()).decode("ascii")
    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return {
        "captured_at": captured_at,
        "sample_rate_hz": sample_rate_hz,
        "num_samples": int(len(window_f32)),
        "axis": axis,
        "source_dataset": source_dataset,
        "samples_b64": samples_b64,
    }


def build_status_payload(status: str, rpm: int) -> dict:
    last_seen = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return {"status": status, "rpm": rpm, "last_seen": last_seen}


def select_source(sensor_cfg: dict, inject_fault_label: str | None, fault_type: str) -> dict:
    """Pick which .mat file + nominal_rpm to replay: healthy by default,
    or the configured fault file if --inject-fault matches this sensor's
    bearing_position_label."""
    if inject_fault_label is None:
        h = sensor_cfg["healthy"]
        return {
            "condition": "HEALTHY",
            "file": h["file"],
            "mat_id": h["mat_id"],
            "nominal_rpm": h["nominal_rpm"],
        }

    if inject_fault_label != sensor_cfg["bearing_position_label"]:
        raise ValueError(
            f"--inject-fault '{inject_fault_label}' does not match this sensor's "
            f"bearing_position_label '{sensor_cfg['bearing_position_label']}' "
            f"(only one sensor configured in this session's config)."
        )

    faults = sensor_cfg["faults"]
    if fault_type not in faults:
        raise ValueError(f"--fault-type '{fault_type}' not in config faults: {list(faults.keys())}")

    f = faults[fault_type]
    return {
        "condition": fault_type,
        "file": f["file"],
        "mat_id": f["mat_id"],
        "nominal_rpm": f["nominal_rpm"],
    }


def heartbeat_loop(client: mqtt.Client, status_topic: str, rpm: int, stop_event: threading.Event):
    while not stop_event.is_set():
        payload = build_status_payload("ONLINE", rpm)
        client.publish(status_topic, json.dumps(payload), qos=1, retain=True)
        print(f"[heartbeat] Published ONLINE rpm={rpm} to '{status_topic}'")
        stop_event.wait(HEARTBEAT_INTERVAL_S)


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    parser = argparse.ArgumentParser(description="PredictMaint MQTT dataset-replay simulator")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config" / "simulator_config.json"))
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--speed-multiplier", type=float, default=1.0,
                         help="Replay speed factor. 1.0 = real-time (1 window/1.0s). "
                              "10-50 recommended for fast verification, not real demo.")
    parser.add_argument("--inject-fault", metavar="BEARING_POSITION_LABEL", default=None,
                         help="Switch the matching sensor from healthy to faulty source file playback.")
    parser.add_argument("--fault-type", default="OUTER_RACE",
                         choices=["BALL", "INNER_RACE", "OUTER_RACE"],
                         help="Fault class to play when --inject-fault is set. Default: OUTER_RACE.")
    args = parser.parse_args()

    if args.speed_multiplier <= 0:
        raise ValueError("--speed-multiplier must be > 0")

    env_file = load_env(Path(args.env_file))
    host = get_setting(env_file, "MQTT_BROKER_HOST")
    port = int(get_setting(env_file, "MQTT_BROKER_PORT"))
    username = get_setting(env_file, "MQTT_BROKER_USERNAME")
    password = get_setting(env_file, "MQTT_BROKER_PASSWORD")

    config = json.loads(Path(args.config).read_text())
    sensor_cfg = config["sensors"][0]

    sample_rate_hz = sensor_cfg["sample_rate_hz"]
    window_samples = sample_rate_hz
    axis = sensor_cfg["sensor_axis"]
    source_dataset = sensor_cfg["source_dataset"]
    vibration_topic = sensor_cfg["mqtt_topic_vibration"]
    status_topic = sensor_cfg["mqtt_topic_status"]

    source = select_source(sensor_cfg, args.inject_fault, args.fault_type)
    mat_path = Path(sensor_cfg["dataset_root"]) / source["file"]
    samples = read_de_channel(mat_path, source["mat_id"])
    n_windows, dropped = chunk_plan(samples, window_samples)

    print(f"[simulator] Condition: {source['condition']} | File: {mat_path.name} | "
          f"RPM: {source['nominal_rpm']} | speed_multiplier={args.speed_multiplier}")
    print(f"[simulator] {len(samples)} samples -> {n_windows} complete {window_samples}-sample windows "
          f"({dropped} trailing samples dropped)")

    lwt_payload = json.dumps(build_status_payload("OFFLINE", source["nominal_rpm"]))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="predictmaint-simulator-drive-end")
    client.username_pw_set(username, password)
    client.will_set(status_topic, payload=lwt_payload, qos=1, retain=True)
    client.connect(host, port)
    client.loop_start()
    print(f"[simulator] Connected to {host}:{port} as '{username}'. LWT armed on '{status_topic}' "
          f"(note: LWT payload's last_seen reflects connect time, not actual disconnect time — "
          f"this is inherent to MQTT LWT since the client cannot update it after going offline; "
          f"real-time offline detection should rely on message arrival gaps, not this field, per "
          f"mqtt-architecture.md §Sensor Offline Detection)")

    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=heartbeat_loop,
        args=(client, status_topic, source["nominal_rpm"], stop_event),
        daemon=True,
    )
    hb_thread.start()

    interval_s = 1.0 / args.speed_multiplier
    idx = 0
    try:
        while True:
            wrap = idx // n_windows
            start = (idx % n_windows) * window_samples
            if idx % n_windows == 0 and wrap > 0:
                print(f"[simulator] Reached end of source file, wrapping to start (loop #{wrap})")
            window = samples[start:start + window_samples]
            payload = build_vibration_payload(window, sample_rate_hz, axis, source_dataset)
            client.publish(vibration_topic, json.dumps(payload), qos=1)
            print(f"[simulator] Published window {idx} (source samples {start}:{start + window_samples}) "
                  f"captured_at={payload['captured_at']}")
            idx += 1
            time.sleep(interval_s)
    except (KeyboardInterrupt, GracefulExit) as e:
        reason = "Ctrl+C" if isinstance(e, KeyboardInterrupt) else "SIGTERM (kill)"
        print(f"\n[simulator] Stopped ({reason}) — publishing graceful OFFLINE before disconnect")
    finally:
        stop_event.set()
        # Graceful shutdown publishes explicit OFFLINE with a FRESH
        # timestamp (unlike the LWT payload, which is fixed at connect
        # time). This now covers both Ctrl+C and `kill` (SIGTERM).
        client.publish(status_topic, json.dumps(build_status_payload("OFFLINE", source["nominal_rpm"])),
                        qos=1, retain=True)
        time.sleep(0.2)  # give the publish a moment to flush before disconnect
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
