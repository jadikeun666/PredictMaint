"""
`python manage.py run_ingestion`

Subscriber MQTT generik (architecture.md §Services — ingestion-service,
"thin ingestion, fat worker" per engineering-rules.md). Berlangganan ke
SEMUA topic `factory/+/+/vibration` dan `factory/+/+/status` (wildcard).

Cakupan (giliran 1-4, PredictMaint):
- Validasi payload PERSIS skema mqtt-architecture.md
- Resolusi sensor_id dari topic (read-only, registry = source of truth)
- Decode samples_b64 -> .npy + insert vibration_windows (immutable)
- Idempotency (sensor_id, captured_at) — mqtt-architecture.md §QoS Policy
- Dead-letter log persisten (JSONL) untuk payload malformed/invalid
- Enqueue Celery task ke queue signal_processing (string task name saja,
  TIDAK import apps.signal_processing — engineering-rules.md queue
  boundary, BATASAN sesi ini)
- SENSOR_OFFLINE detection berbasis waktu KEDATANGAN pesan (bukan field
  last_seen di payload — mqtt-architecture.md Addendum 2026-08-16) +
  Alert dengan cooldown pattern + auto-resolve saat sensor online lagi
"""
import json
import logging
import signal
import time
import uuid
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.utils.dateparse import parse_datetime
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from apps.ingestion.validation import validate_vibration_payload, validate_status_payload
from apps.ingestion.services import (
    ingest_vibration_window,
    find_existing_window,
    write_dead_letter,
    upsert_sensor_offline_alert,
    resolve_sensor_offline_alert,
)
from apps.registry.models import Sensor
from config.celery import app as celery_app

logger = logging.getLogger("ingestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

VIBRATION_TOPIC_FILTER = "factory/+/+/vibration"
STATUS_TOPIC_FILTER = "factory/+/+/status"

# mqtt-architecture.md §Sensor Offline Detection: heartbeat tiap 30s,
# threshold offline = 3x interval = 90s.
HEARTBEAT_INTERVAL_S = 30
OFFLINE_THRESHOLD_S = HEARTBEAT_INTERVAL_S * 3
# Resolusi polling loop offline-check — keputusan implementasi, bukan
# nilai dari spec manapun.
CHECK_INTERVAL_S = 15


class GracefulExit(Exception):
    pass


def _sigterm_handler(signum, frame):
    raise GracefulExit()


def _resolve_vibration_topic(status_topic):
    """factory/{slug}/{position}/status -> factory/{slug}/{position}/vibration"""
    parts = status_topic.split("/")
    if len(parts) != 4 or parts[0] != "factory" or parts[3] != "status":
        return None
    return "/".join(parts[:3] + ["vibration"])


class Command(BaseCommand):
    help = "MQTT ingestion subscriber (validasi, persist, idempotency, dead-letter, enqueue, SENSOR_OFFLINE)."

    def handle(self, *args, **options):
        signal.signal(signal.SIGTERM, _sigterm_handler)

        # sensor_id -> (Sensor, datetime kedatangan pesan status terakhir)
        self._last_status_seen = {}

        username = getattr(settings, "MQTT_BROKER_USERNAME", "") or None
        password = getattr(settings, "MQTT_BROKER_PASSWORD", "") or None
        host = settings.MQTT_BROKER_HOST
        port = int(settings.MQTT_BROKER_PORT)

        client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=f"ingestion-service-{uuid.uuid4().hex[:8]}")
        if username:
            client.username_pw_set(username, password)

        client.on_connect = self._on_connect
        client.on_message = self._on_message

        logger.info(f"Connecting to MQTT broker {host}:{port} as ingestion-service...")
        client.connect(host, port, keepalive=60)
        client.loop_start()

        try:
            while True:
                time.sleep(CHECK_INTERVAL_S)
                self._check_offline_sensors()
        except (KeyboardInterrupt, GracefulExit) as e:
            reason = "Ctrl+C" if isinstance(e, KeyboardInterrupt) else "SIGTERM (docker stop)"
            logger.info(f"Shutting down ingestion-service ({reason})...")
        finally:
            client.loop_stop()
            client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info(f"Connected. Subscribing to {VIBRATION_TOPIC_FILTER} and {STATUS_TOPIC_FILTER}")
            client.subscribe([(VIBRATION_TOPIC_FILTER, 1), (STATUS_TOPIC_FILTER, 1)])
        else:
            logger.error(f"Connection failed: reason_code={reason_code}")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            path = write_dead_letter(topic, msg.payload, [f"malformed JSON: {exc}"])
            logger.warning(f"[DEAD-LETTER] topic={topic} — malformed JSON payload, ditulis ke {path}")
            return

        if topic.endswith("/vibration"):
            self._handle_vibration(topic, payload, msg.payload)
        elif topic.endswith("/status"):
            self._handle_status(topic, payload, msg.payload)
        else:
            logger.warning(f"[ANOMALY] message on unexpected topic shape: {topic}")

    def _handle_vibration(self, topic, payload, msg_payload_raw):
        is_valid, errors = validate_vibration_payload(payload)
        if not is_valid:
            path = write_dead_letter(topic, msg_payload_raw, errors)
            logger.warning(f"[DEAD-LETTER] topic={topic} — invalid vibration payload, ditulis ke {path}: {errors}")
            return

        sensor = Sensor.objects.filter(mqtt_topic=topic).select_related("bearing__machine").first()
        if sensor is None:
            logger.warning(
                f"[ANOMALY] valid vibration payload on topic={topic} but no registered Sensor.mqtt_topic "
                f"matches — registry is source of truth, not auto-creating"
            )
            return

        captured_at_dt = parse_datetime(payload["captured_at"])
        existing = find_existing_window(sensor, captured_at_dt)
        if existing is not None:
            logger.info(
                f"[DUPLICATE] topic={topic} sensor_id={sensor.id} captured_at={payload['captured_at']} "
                f"— window sudah ada (window_id={existing.id}), skip insert & enqueue (idempotency)"
            )
            return

        try:
            window = ingest_vibration_window(sensor, payload)
        except Exception:
            logger.exception(
                f"[INGEST ERROR] topic={topic} sensor_id={sensor.id} — gagal decode/tulis .npy "
                f"atau insert vibration_windows (payload sudah valid, ini kegagalan infra, bukan dead-letter)"
            )
            return

        logger.info(
            f"[OK] vibration window persisted — topic={topic} sensor_id={sensor.id} "
            f"machine={sensor.bearing.machine.name} bearing={sensor.bearing.position_label} "
            f"window_id={window.id} captured_at={payload['captured_at']} "
            f"num_samples={payload['num_samples']} duration_s={window.duration_s} "
            f"waveform_path={window.waveform_path}"
        )

        try:
            celery_app.send_task(
                "apps.signal_processing.tasks.process_vibration_window",
                kwargs={"window_id": str(window.id)},
                queue="signal_processing",
            )
            logger.info(f"[ENQUEUE] window_id={window.id} -> queue=signal_processing")
        except Exception:
            logger.exception(f"[ENQUEUE ERROR] window_id={window.id} — gagal enqueue ke Celery")

    def _handle_status(self, topic, payload, msg_payload_raw):
        is_valid, errors = validate_status_payload(payload)
        if not is_valid:
            path = write_dead_letter(topic, msg_payload_raw, errors)
            logger.warning(f"[DEAD-LETTER] topic={topic} — invalid status payload, ditulis ke {path}: {errors}")
            return

        vibration_topic = _resolve_vibration_topic(topic)
        sensor = (
            Sensor.objects.filter(mqtt_topic=vibration_topic).select_related("bearing__machine").first()
            if vibration_topic else None
        )
        if sensor is None:
            logger.warning(f"[ANOMALY] valid status payload on topic={topic} but no registered Sensor matches")
            return

        arrival_time = timezone.now()
        self._last_status_seen[sensor.id] = (sensor, arrival_time)

        resolved_count = resolve_sensor_offline_alert(sensor)
        if resolved_count:
            logger.info(f"[SENSOR_OFFLINE] sensor_id={sensor.id} kembali online — alert auto-resolved")

        logger.info(
            f"[OK] status heartbeat valid — topic={topic} sensor_id={sensor.id} "
            f"status={payload['status']} rpm={payload['rpm']}"
        )

    def _check_offline_sensors(self):
        now = timezone.now()
        for sensor_id, (sensor, last_seen) in list(self._last_status_seen.items()):
            if now - last_seen > timedelta(seconds=OFFLINE_THRESHOLD_S):
                alert, created = upsert_sensor_offline_alert(sensor)
                if created:
                    logger.warning(
                        f"[SENSOR_OFFLINE] alert_id={alert.id} sensor_id={sensor.id} bearing={sensor.bearing.position_label} "
                        f"— tidak ada pesan status selama >{OFFLINE_THRESHOLD_S}s (last_seen={last_seen.isoformat()})"
                    )
                else:
                    logger.info(f"[SENSOR_OFFLINE] cooldown update alert_id={alert.id} sensor_id={sensor.id}")
