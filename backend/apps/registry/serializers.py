import re

from rest_framework import serializers

from .models import Bearing, Machine, Sensor
from .topics import build_vibration_topic, slugify_machine_name

_POSITION_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class MachineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Machine
        fields = [
            "id",
            "name",
            "location",
            "machine_type",
            "criticality",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]

    def validate_name(self, value):
        new_slug = slugify_machine_name(value)
        if not new_slug:
            raise serializers.ValidationError(
                "Nama harus mengandung minimal satu huruf/angka (dipakai membentuk MQTT topic)."
            )
        if self.instance is not None:
            old_slug = slugify_machine_name(self.instance.name)
            if new_slug != old_slug and Sensor.objects.filter(
                bearing__machine=self.instance
            ).exists():
                raise serializers.ValidationError(
                    "Nama tidak bisa diubah (slug MQTT berubah) selama mesin ini masih "
                    "punya sensor — topic sensor akan tidak cocok dengan publisher/ingestion."
                )
        return value


class BearingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bearing
        fields = [
            "id",
            "machine",
            "position_label",
            "bearing_model",
            "ball_diameter_mm",
            "pitch_diameter_mm",
            "contact_angle_deg",
            "num_balls",
            "installed_at",
            "status",
        ]
        read_only_fields = ["id"]

    def validate_position_label(self, value):
        if not _POSITION_LABEL_RE.match(value):
            raise serializers.ValidationError(
                "Hanya huruf, angka, '.', '_' dan '-' (dipakai sebagai segmen MQTT topic)."
            )
        return value

    def validate_ball_diameter_mm(self, value):
        if value is not None and value <= 0:
            raise serializers.ValidationError("Harus > 0.")
        return value

    def validate_pitch_diameter_mm(self, value):
        if value is not None and value <= 0:
            raise serializers.ValidationError("Harus > 0.")
        return value

    def validate_num_balls(self, value):
        if value is not None and value <= 0:
            raise serializers.ValidationError("Harus > 0.")
        return value

    def validate(self, attrs):
        inst = self.instance

        def pick(field):
            if field in attrs:
                return attrs[field]
            return getattr(inst, field, None) if inst is not None else None

        machine = pick("machine")
        label = pick("position_label")

        d = pick("ball_diameter_mm")
        D = pick("pitch_diameter_mm")
        if d is not None and D is not None and d >= D:
            raise serializers.ValidationError(
                {"ball_diameter_mm": "Harus lebih kecil dari pitch_diameter_mm (d < D)."}
            )

        if machine is not None and label is not None:
            clash = Bearing.objects.filter(machine=machine, position_label=label)
            if inst is not None:
                clash = clash.exclude(pk=inst.pk)
            if clash.exists():
                raise serializers.ValidationError(
                    {"position_label": "Sudah dipakai bearing lain pada mesin ini."}
                )

        if inst is not None and Sensor.objects.filter(bearing=inst).exists():
            if ("machine" in attrs and attrs["machine"] != inst.machine) or (
                "position_label" in attrs and attrs["position_label"] != inst.position_label
            ):
                raise serializers.ValidationError(
                    "machine/position_label tidak bisa diubah selama bearing ini masih punya "
                    "sensor — mqtt_topic sensor diturunkan dari keduanya."
                )
        return attrs


class SensorSerializer(serializers.ModelSerializer):
    """
    mqtt_topic read-only: digenerate dari machine.name (slug) +
    bearing.position_label sesuai mqtt-architecture.md §Topic Structure,
    supaya operator tidak bisa salah ketik topic dan merusak routing MQTT.
    `bearing` hanya bisa diisi saat create (topic diturunkan darinya).
    """

    class Meta:
        model = Sensor
        fields = [
            "id",
            "bearing",
            "sensor_type",
            "axis",
            "sample_rate_hz",
            "source",
            "mqtt_topic",
        ]
        read_only_fields = ["id", "mqtt_topic"]

    def validate_sample_rate_hz(self, value):
        if value is not None and value <= 0:
            raise serializers.ValidationError("Harus > 0.")
        return value

    def validate(self, attrs):
        inst = self.instance
        if inst is not None:
            if "bearing" in attrs and attrs["bearing"] != inst.bearing:
                raise serializers.ValidationError(
                    {"bearing": "Tidak bisa dipindah ke bearing lain (mqtt_topic diturunkan dari bearing)."}
                )
            return attrs

        bearing = attrs["bearing"]
        if not slugify_machine_name(bearing.machine.name):
            raise serializers.ValidationError(
                {"bearing": "Nama mesin tidak menghasilkan slug MQTT yang valid."}
            )
        topic = build_vibration_topic(bearing.machine.name, bearing.position_label)
        if Sensor.objects.filter(mqtt_topic=topic).exists():
            raise serializers.ValidationError(
                {
                    "bearing": (
                        f"Sudah ada sensor dengan mqtt_topic '{topic}'. Topic tidak memuat axis, "
                        "jadi satu bearing hanya bisa punya satu sensor."
                    )
                }
            )
        attrs["mqtt_topic"] = topic
        return attrs
