"""
Seed minimal registry data (Machine, Bearing, Sensor) needed for the
MQTT dataset-replay simulator to have a target machine/bearing/sensor
to publish against.

Geometry values copied VERBATIM from bearing-fault-formulas.md
§Known Geometry Values (CWRU / SKF 6205-2RS JEM, drive end).

Idempotent: safe to re-run, uses get_or_create keyed on natural
identifiers (machine name, bearing position_label + machine, sensor
mqtt_topic).
"""
from django.core.management.base import BaseCommand
from django.utils.text import slugify

from apps.registry.models import Machine, Bearing, Sensor


class Command(BaseCommand):
    help = "Seed minimal registry data for the MQTT simulator (CWRU drive-end bearing)."

    def handle(self, *args, **options):
        machine_name = "CWRU Test Rig 1"
        machine_slug = slugify(machine_name)  # -> "cwru-test-rig-1"
        bearing_position_label = "drive-end"

        machine, m_created = Machine.objects.get_or_create(
            name=machine_name,
            defaults={
                "location": "CWRU Bearing Data Center (dataset test bench)",
                "machine_type": "bearing_test_rig",
                "criticality": Machine.Criticality.HIGH,
            },
        )
        self.stdout.write(
            self.style.SUCCESS(f"Machine {'created' if m_created else 'exists'}: {machine.id} ({machine.name})")
        )

        bearing, b_created = Bearing.objects.get_or_create(
            machine=machine,
            position_label=bearing_position_label,
            defaults={
                # PERSIS dari bearing-fault-formulas.md §Known Geometry Values
                "bearing_model": "SKF 6205-2RS JEM (drive end)",
                "ball_diameter_mm": 7.94,
                "pitch_diameter_mm": 39.04,
                "contact_angle_deg": 0,
                "num_balls": 9,
                "status": Bearing.Status.ACTIVE,
            },
        )
        self.stdout.write(
            self.style.SUCCESS(f"Bearing {'created' if b_created else 'exists'}: {bearing.id} ({bearing.bearing_model})")
        )

        mqtt_topic = f"factory/{machine_slug}/{bearing_position_label}/vibration"

        sensor, s_created = Sensor.objects.get_or_create(
            bearing=bearing,
            axis=Sensor.Axis.RADIAL,
            defaults={
                "sensor_type": Sensor.SensorType.ACCELEROMETER,
                # ASUMSI DITANDAI: 48000 Hz berdasarkan nama folder dataset
                # `CWRU_48k_load_1_CNN_data` di claude.md §Datasets — belum
                # diverifikasi terhadap metadata internal file .mat yang
                # sebenarnya. WAJIB dikonfirmasi ulang saat script simulator
                # (giliran 2) benar-benar membaca file dan membandingkan
                # sample_rate dari metadata .mat, bukan diasumsikan permanen.
                "sample_rate_hz": 48000,
                "source": Sensor.Source.SIMULATED,
                "mqtt_topic": mqtt_topic,
            },
        )
        self.stdout.write(
            self.style.SUCCESS(f"Sensor {'created' if s_created else 'exists'}: {sensor.id} (topic={sensor.mqtt_topic})")
        )

        self.stdout.write(self.style.SUCCESS("Seed complete."))
