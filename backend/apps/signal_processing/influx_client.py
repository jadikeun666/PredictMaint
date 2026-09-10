"""
Shared InfluxDB client helpers for celery-worker-signal-processing.

Measurements touched from this app (database.md §InfluxDB Schema, plus
Giliran 3's addendum):
  - `derived_features` — WRITTEN here (Giliran 5, step 9).
  - `sensor_status`    — Giliran 3 addendum. WRITTEN by rpm-listener,
                         READ here for `fr` (step 7).
  - `predictions_ts`   — NOT touched here; celery-worker-ml-inference's job.
"""
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from django.conf import settings

SENSOR_STATUS_MEASUREMENT = "sensor_status"
DERIVED_FEATURES_MEASUREMENT = "derived_features"

DERIVED_FEATURES_FIELD_NAMES = (
    "rms", "kurtosis", "crest_factor", "skewness", "dominant_freq_hz",
    "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp",
)


def get_influx_client() -> InfluxDBClient:
    return InfluxDBClient(
        url=settings.INFLUXDB_URL,
        token=settings.INFLUXDB_TOKEN,
        org=settings.INFLUXDB_ORG,
    )


def write_derived_features(machine_id, bearing_id, sensor_id, captured_at, features: dict):
    """
    Step 9: write one derived_features point.
    Tags: machine_id, bearing_id, sensor_id (database.md).
    Timestamp: captured_at of the SOURCE vibration_windows row (not
    "now") — so charts/exports align with when the physical vibration
    actually happened, not when the worker happened to process it.
    """
    client = get_influx_client()
    try:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        point = (
            Point(DERIVED_FEATURES_MEASUREMENT)
            .tag("machine_id", str(machine_id))
            .tag("bearing_id", str(bearing_id))
            .tag("sensor_id", str(sensor_id))
            .time(captured_at, WritePrecision.NS)
        )
        for field_name in DERIVED_FEATURES_FIELD_NAMES:
            value = features.get(field_name)
            if value is not None:
                point = point.field(field_name, float(value))
        write_api.write(bucket=settings.INFLUXDB_BUCKET, record=point)
    finally:
        client.close()


def get_latest_rpm_at_or_before(sensor_id, at_timestamp):
    """
    Query `sensor_status` for the most recent `rpm` field value at or
    before `at_timestamp` for the given sensor_id. Returns float(rpm) or
    None if no heartbeat exists at/before that time.
    """
    client = get_influx_client()
    try:
        query_api = client.query_api()
        at_iso = at_timestamp.isoformat()
        flux = f'''
        from(bucket: "{settings.INFLUXDB_BUCKET}")
          |> range(start: 0, stop: {at_iso})
          |> filter(fn: (r) => r._measurement == "{SENSOR_STATUS_MEASUREMENT}")
          |> filter(fn: (r) => r.sensor_id == "{sensor_id}")
          |> filter(fn: (r) => r._field == "rpm")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 1)
        '''
        tables = query_api.query(flux)
        for table in tables:
            for record in table.records:
                return float(record.get_value())
        return None
    finally:
        client.close()
