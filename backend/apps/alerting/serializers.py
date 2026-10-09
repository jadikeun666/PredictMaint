from rest_framework import serializers

from .models import Alert


class AlertSerializer(serializers.ModelSerializer):
    """api.md: nama field persis database.md. Semua read-only — alert dibuat sistem;
    satu-satunya mutasi manual adalah POST /alerts/{id}/resolve/."""

    class Meta:
        model = Alert
        fields = [
            "id",
            "bearing",
            "spare_part",
            "alert_type",
            "severity",
            "related_prediction",
            "triggered_at",
            "last_seen_at",
            "resolved_at",
            "resolved_by",
        ]
        read_only_fields = fields
