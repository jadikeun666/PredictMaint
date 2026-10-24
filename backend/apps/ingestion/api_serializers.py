from rest_framework import serializers

from .models import VibrationWindow


class VibrationWindowSerializer(serializers.ModelSerializer):
    """Nama field persis database.md; seluruhnya read-only (immutable, system-only)."""

    class Meta:
        model = VibrationWindow
        fields = [
            "id",
            "sensor",
            "captured_at",
            "duration_s",
            "num_samples",
            "waveform_path",
            "source_dataset",
            "ingested_at",
        ]
        read_only_fields = fields
