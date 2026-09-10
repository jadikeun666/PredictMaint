import uuid

from django.db import models


class Machine(models.Model):
    class Criticality(models.TextChoices):
        LOW = "LOW", "Low"
        MEDIUM = "MEDIUM", "Medium"
        HIGH = "HIGH", "High"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255, blank=True)
    machine_type = models.CharField(max_length=100)
    criticality = models.CharField(max_length=10, choices=Criticality.choices)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "machines"

    def __str__(self):
        return self.name


class Bearing(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE", "Active"
        RETIRED = "RETIRED", "Retired"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    machine = models.ForeignKey(Machine, on_delete=models.PROTECT, related_name="bearings")
    position_label = models.CharField(max_length=100)
    bearing_model = models.CharField(max_length=255)
    # Nullable at creation; required before BPFO/BPFI/BSF/FTF features can be
    # computed (prd.md Epic A2) — enforced at application/serializer layer,
    # not at the DB level, per bearing-fault-formulas.md.
    ball_diameter_mm = models.DecimalField(max_digits=6, decimal_places=3, null=True, blank=True)
    pitch_diameter_mm = models.DecimalField(max_digits=6, decimal_places=3, null=True, blank=True)
    contact_angle_deg = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    num_balls = models.IntegerField(null=True, blank=True)
    installed_at = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.ACTIVE)

    class Meta:
        db_table = "bearings"

    def __str__(self):
        return f"{self.machine.name} / {self.position_label}"


class Sensor(models.Model):
    class SensorType(models.TextChoices):
        ACCELEROMETER = "ACCELEROMETER", "Accelerometer"

    class Axis(models.TextChoices):
        RADIAL = "RADIAL", "Radial"
        AXIAL = "AXIAL", "Axial"

    class Source(models.TextChoices):
        SIMULATED = "SIMULATED", "Simulated"
        HARDWARE = "HARDWARE", "Hardware"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bearing = models.ForeignKey(Bearing, on_delete=models.PROTECT, related_name="sensors")
    sensor_type = models.CharField(max_length=20, choices=SensorType.choices, default=SensorType.ACCELEROMETER)
    axis = models.CharField(max_length=10, choices=Axis.choices)
    # Read from source file/dataset metadata at ingestion time, never
    # hardcoded — see signal-processing.md §Dataset-Specific Ingestion Notes.
    sample_rate_hz = models.IntegerField()
    source = models.CharField(max_length=10, choices=Source.choices, default=Source.SIMULATED)
    mqtt_topic = models.CharField(max_length=255, unique=True)

    class Meta:
        db_table = "sensors"

    def __str__(self):
        return self.mqtt_topic
