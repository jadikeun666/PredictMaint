from rest_framework import serializers

from .models import Prediction


class PredictionSerializer(serializers.ModelSerializer):
    """Nama field persis database.md (api.md melarang rename). Seluruhnya read-only —
    predictions dibuat sistem oleh worker ml_inference, tanpa endpoint tulis manual."""

    class Meta:
        model = Prediction
        fields = [
            "id",
            "window",
            "bearing",
            "prediction_type",
            "fault_class",
            "classification_confidence",
            "predicted_rul_hours",
            "rul_confidence_band_low_hours",
            "rul_confidence_band_high_hours",
            "model_version",
            "predicted_at",
        ]
        read_only_fields = fields
