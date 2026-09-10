"""
Standalone MQTT listener that persists live RPM (from `.../status`
heartbeats) into InfluxDB, measurement `sensor_status`.

WHY THIS EXISTS (Giliran-0.5 design decision from this session's chat):
- bearing-fault-formulas.md / mqtt-architecture.md require `fr` (shaft
  rotation frequency) to be read from the live RPM heartbeat at
  feature-computation time — not assumed constant.
- The `Sensor` model (apps/registry/models.py) has NO rpm column, and
  ingestion-service only holds the last-seen RPM in an in-process dict
  (self._last_status_seen), never persisted anywhere
  celery-worker-signal-processing can read.
- BATASAN 1 this session forbids touching apps/ingestion at all;
  BATASAN 4 forbids a Postgres schema change without explicit approval.
- Chosen fix (user-approved, chosen for industry-standard precision):
  a NEW, independent MQTT subscriber that ONLY reads `.../status`
  (read-only — MQTT supports multiple subscribers on the same topic)
  and writes each RPM reading as a time-series point to InfluxDB. This
  keeps a full RPM history (not just "last known value"), so
  fault-frequency extraction can look up the RPM closest to a specific
  window's captured_at rather than an approximate current snapshot.
  New InfluxDB measurement only — no Postgres migration.

Run as its own process / Docker Compose service (`rpm-listener`),
independent of both ingestion-service and any Celery worker process.
Deliberately does NOT do SENSOR_OFFLINE detection (still exclusively
ingestion-service's job, per architecture.md) — this listener has one
job only: capture rpm -> InfluxDB.
"""
import json
import logging
import signal
import uuid

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.dateparse import parse_datetime
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from apps.registry.models import Sensor
from apps.signal_processing.influx_client import get_influx_client, SENSOR_STATUS_MEASUREMENT

logger = logging.getLogger("rpm_listener")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STATUS_TOPIC_FILTER = "factory/+/+/status"


class GracefulExit(Exception):
    pass


def _sigterm_handler(signum, frame):
    raise GracefulExit()


def _status_topic_to_vibration_topic(status_topic):
    """factory/{slug}/{position}/status -> factory/{slug}/{position}/vibration

    NOTE: intentionally duplicated (NOT imported) from apps/ingestion's
    private _resolve_vibration_topic — this listener is deliberately
    independent of apps/ingestion per BATASAN 1 (no import from
    apps.ingestion at all this session). This is a trivial 4-line
    string transform, not a domain "formula" subject to
    engineering-rules.md's shared-utility-module rule (that rule targets
    things like BPFO/BPFI, not topic-string plumbing) — so duplicating
    it here is a deliberate, documented trade-off, not an oversight.
    """
    parts = status_topic.split("/")
    if len(parts) != 4 or parts[0] != "factory" or parts[3] != "status":
        return None
    return "/".join(parts[:3] + ["vibration"])


class Command(BaseCommand):
    help = "Standalone MQTT listener: persists live RPM from .../status heartbeats into InfluxDB (sensor_status measurement)."

    def handle(self, *args, **options):
        signal.signal(signal.SIGTERM, _sigterm_handler)

        self.influx_client = get_influx_client()
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

        username = getattr(settings, "MQTT_BROKER_USERNAME", "") or None
        password = getattr(settings, "MQTT_BROKER_PASSWORD", "") or None
        host = settings.MQTT_BROKER_HOST
        port = int(settings.MQTT_BROKER_PORT)

        client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=f"rpm-listener-{uuid.uuid4().hex[:8]}")
        if username:
            client.username_pw_set(username, password)

        client.on_connect = self._on_connect
        client.on_message = self._on_message

        logger.info(f"Connecting to MQTT broker {host}:{port} as rpm-listener...")
        client.connect(host, port, keepalive=60)
        try:
            client.loop_forever()
        except (KeyboardInterrupt, GracefulExit) as e:
            reason = "Ctrl+C" if isinstance(e, KeyboardInterrupt) else "SIGTERM (docker stop)"
            logger.info(f"Shutting down rpm-listener ({reason})...")
        finally:
            client.disconnect()
            self.influx_client.close()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info(f"Connected. Subscribing to {STATUS_TOPIC_FILTER} (read-only, RPM capture)")
            client.subscribe([(STATUS_TOPIC_FILTER, 1)])
        else:
            logger.error(f"Connection failed: reason_code={reason_code}")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning(f"[SKIP] topic={topic} — malformed JSON: {exc} (dead-lettering stays ingestion-service's job, not duplicated here)")
            return

        rpm = payload.get("rpm")
        last_seen_raw = payload.get("last_seen")
        if rpm is None or last_seen_raw is None:
            logger.debug(f"[SKIP] topic={topic} — no rpm/last_seen (e.g. OFFLINE/LWT message): {payload}")
            return

        last_seen_dt = parse_datetime(last_seen_raw)
        if last_seen_dt is None:
            logger.warning(f"[SKIP] topic={topic} — unparseable last_seen={last_seen_raw!r}")
            return

        vibration_topic = _status_topic_to_vibration_topic(topic)
        sensor = (
            Sensor.objects.filter(mqtt_topic=vibration_topic).select_related("bearing__machine").first()
            if vibration_topic else None
        )
        if sensor is None:
            logger.warning(f"[ANOMALY] status payload on topic={topic} but no registered Sensor matches — skipping RPM write")
            return

        point = (
            Point(SENSOR_STATUS_MEASUREMENT)
            .tag("sensor_id", str(sensor.id))
            .tag("bearing_id", str(sensor.bearing_id))
            .tag("machine_id", str(sensor.bearing.machine_id))
            .field("rpm", float(rpm))
            .time(last_seen_dt, WritePrecision.NS)
        )
        try:
            self.write_api.write(bucket=settings.INFLUXDB_BUCKET, record=point)
            logger.info(f"[OK] rpm={rpm} sensor_id={sensor.id} bearing={sensor.bearing.position_label} last_seen={last_seen_raw}")
        except Exception:
            logger.exception(f"[INFLUX WRITE ERROR] sensor_id={sensor.id} rpm={rpm}")
