#!/usr/bin/env python3
"""
PredictMaint — IMS run-to-failure replay publisher (Sesi C, Giliran 3)

Skrip standalone (bukan bagian aplikasi Django). Membaca snapshot ASCII IMS satu
bearing, mempublikasikan tiap snapshot sebagai satu window ke topik
factory/{machine_slug}/{bearing_label}/vibration dengan kontrak payload
mqtt-architecture.md (source_dataset "IMS", 20.480 sampel @ 20 kHz) + heartbeat
.../status (RPM 2000, konstan menurut Readme IMS).

Memakai ulang helper simulator/publisher.py (TIDAK diubah): payload, heartbeat,
LWT, penanganan SIGTERM.

BATASAN (didokumentasikan): worker signal_processing memproses beberapa window
paralel. Pada replay cepat, window dapat selesai tidak berurutan sehingga jendela
6 titik RUL bisa terbentuk dari titik tidak berurutan. Untuk verifikasi gunakan
--speed-multiplier <= 2. Waktu replay terkompresi: RUL dalam "jam waktu dataset".

Contoh:
  python3 simulator/replay_ims.py --traj IMS-S2-B1 --limit 60 --speed-multiplier 2
  python3 simulator/replay_ims.py --traj IMS-S2-B1 --dry-run
"""
import argparse
import json
import re
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

import publisher as pub   # helper reuse; publisher.py tidak diubah

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Fakta dari Readme IMS: 20 kHz, 20.480 titik/snapshot, 2000 RPM konstan.
IMS_SAMPLE_RATE_HZ = 20000
IMS_SNAPSHOT_SAMPLES = 20480
IMS_RPM = 2000
DEFAULT_IMS_ROOT = "/mnt/c/libursem4/PredictMaint/dataset/IMS Bearing Data"
DEFAULT_MANIFEST = Path("/tmp/ims_replay_manifest.json")

# kolom = indeks 0-based kanal primer (sama dengan ims_dataset.TRAJECTORIES)
TRAJ = {
    "IMS-S1-B3": dict(folder="1st_test/1st_test", column=4, n_columns=8, machine_slug="ims-set-1", label="bearing-3"),
    "IMS-S1-B4": dict(folder="1st_test/1st_test", column=6, n_columns=8, machine_slug="ims-set-1", label="bearing-4"),
    "IMS-S2-B1": dict(folder="2nd_test/2nd_test", column=0, n_columns=4, machine_slug="ims-set-2", label="bearing-1"),
    "IMS-S3-B3": dict(folder="3rd_test/4th_test/txt", column=2, n_columns=4, machine_slug="ims-set-3", label="bearing-3"),
}
_NAME_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


def list_snapshots(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder snapshot tidak ada: {folder}")
    return sorted((p for p in folder.iterdir() if p.is_file() and _NAME_RE.match(p.name)), key=lambda p: p.name)


def load_window(path: Path, n_columns: int, column: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.shape != (IMS_SNAPSHOT_SAMPLES, n_columns):
        raise ValueError(f"Bentuk {path.name} = {arr.shape}, diharapkan ({IMS_SNAPSHOT_SAMPLES}, {n_columns})")
    return arr[:, column]


def main():
    signal.signal(signal.SIGTERM, pub._handle_sigterm)
    ap = argparse.ArgumentParser(description="PredictMaint IMS replay publisher")
    ap.add_argument("--traj", required=True, choices=sorted(TRAJ))
    ap.add_argument("--ims-root", default=DEFAULT_IMS_ROOT)
    ap.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    ap.add_argument("--speed-multiplier", type=float, default=1.0,
                    help="1.0 = 1 snapshot/detik. Gunakan <= 2 untuk verifikasi (lihat batasan di docstring).")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="hanya N snapshot")
    ap.add_argument("--startup-delay", type=float, default=3.0,
                    help="detik menunggu heartbeat/RPM tercatat sebelum window pertama")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--dry-run", action="store_true", help="hanya cek file/bentuk/topik, tanpa MQTT")
    args = ap.parse_args()
    if args.speed_multiplier <= 0:
        raise ValueError("--speed-multiplier harus > 0")

    cfg = TRAJ[args.traj]
    files = list_snapshots(Path(args.ims_root) / cfg["folder"])[args.start_index:]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise RuntimeError("tidak ada snapshot untuk direplay")
    base = f"factory/{cfg['machine_slug']}/{cfg['label']}"
    vib_topic, status_topic = f"{base}/vibration", f"{base}/status"

    first = load_window(files[0], cfg["n_columns"], cfg["column"])
    print(f"[replay] {args.traj}: {len(files)} snapshot ({files[0].name} .. {files[-1].name}) | kolom {cfg['column']} | "
          f"bentuk window {first.shape} | topik {vib_topic} | RPM {IMS_RPM} | speed {args.speed_multiplier}x")
    if args.dry_run:
        print("[replay] --dry-run: selesai, tidak ada yang dipublikasikan")
        return 0

    env = pub.load_env(Path(args.env_file))
    host = pub.get_setting(env, "MQTT_BROKER_HOST")
    port = int(pub.get_setting(env, "MQTT_BROKER_PORT"))
    user, pwd = pub.get_setting(env, "MQTT_BROKER_USERNAME"), pub.get_setting(env, "MQTT_BROKER_PASSWORD")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"predictmaint-replay-ims-{cfg['label']}-{uuid.uuid4().hex[:8]}")
    client.username_pw_set(user, pwd)
    client.will_set(status_topic, payload=json.dumps(pub.build_status_payload("OFFLINE", IMS_RPM)), qos=1, retain=True)
    client.connect(host, port)
    client.loop_start()
    print(f"[replay] terhubung ke {host}:{port} sebagai '{user}'")

    stop_event = threading.Event()
    threading.Thread(target=pub.heartbeat_loop, args=(client, status_topic, IMS_RPM, stop_event), daemon=True).start()
    time.sleep(args.startup_delay)

    manifest, interval = [], 1.0 / args.speed_multiplier
    t_next = time.monotonic()
    try:
        for i, path in enumerate(files, start=args.start_index):
            window = load_window(path, cfg["n_columns"], cfg["column"])
            payload = pub.build_vibration_payload(window, IMS_SAMPLE_RATE_HZ, "RADIAL", "IMS")
            client.publish(vib_topic, json.dumps(payload), qos=1).wait_for_publish(timeout=10)
            manifest.append({"index": i, "file": path.name, "captured_at": payload["captured_at"]})
            if len(manifest) % 20 == 0 or len(manifest) == len(files):
                print(f"[replay] {len(manifest)}/{len(files)} snapshot dipublikasikan (terakhir {path.name})")
            t_next += interval
            time.sleep(max(0.0, t_next - time.monotonic()))
    except (KeyboardInterrupt, pub.GracefulExit) as e:
        print(f"\n[replay] dihentikan ({'Ctrl+C' if isinstance(e, KeyboardInterrupt) else 'SIGTERM'})")
    finally:
        stop_event.set()
        client.publish(status_topic, json.dumps(pub.build_status_payload("OFFLINE", IMS_RPM)), qos=1, retain=True)
        time.sleep(0.2)
        client.loop_stop()
        client.disconnect()
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps({"traj": args.traj, "topic": vib_topic, "items": manifest}, indent=1))
        print(f"[replay] manifest ({len(manifest)} item) ditulis: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
