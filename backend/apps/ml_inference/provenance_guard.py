"""
apps/ml_inference/provenance_guard.py

Penjaga provenance alert klasifikasi (Sesi E, opsi B; deterministik, bukan deteksi statistik).

Menolak prediksi FAULT berkeyakinan di atas ambang untuk bearing yang `bearing_model`-nya tidak
tervalidasi terhadap `model_version` classifier (config/classification_provenance.py). Penolakan
HANYA mengubah fault_class menjadi "UNKNOWN" (scheduling.md section 1: UNKNOWN -> tanpa alert);
keyakinan mentah TIDAK diubah (tidak ada fabrikasi nilai; kolom classification_confidence wajib
non-null di konsumen tasks._enqueue_alerting dan pdf_report). Penanda penolakan: log WARNING
terstruktur dan konvensi "UNKNOWN dengan keyakinan >= ambang" (mustahil lewat jalur normal, yang
selalu memetakan keyakinan < ambang ke UNKNOWN).

Modul ini RINGAN saat di-import (dipakai tasks.py yang di-autodiscover di semua worker): akses ORM
hanya lazy di dalam lookup_bearing_model, dan hanya dipanggil bila flag menyala dan prediksi
memang layak ditolak (satu query PK).
"""
import logging
from dataclasses import dataclass
from typing import Optional

from config import classification_provenance as P

logger = logging.getLogger(__name__)

# Sama dengan apps.alerting.severity.FAULT_CLASSES (tes memeriksa kesamaannya; tidak diimpor
# karena apps.alerting memuat model Django).
FAULT_CLASSES = frozenset({"INNER_RACE", "OUTER_RACE", "BALL"})


@dataclass(frozen=True)
class ProvenanceDecision:
    rejected: bool
    reason: Optional[str] = None
    bearing_model: Optional[str] = None


NOT_REJECTED = ProvenanceDecision(False)


def lookup_bearing_model(bearing_id) -> Optional[str]:
    """bearing_model dari registry; None bila bearing tidak ada atau kosong."""
    from apps.registry.models import Bearing
    value = Bearing.objects.filter(pk=bearing_id).values_list("bearing_model", flat=True).first()
    return value or None


def check(*, bearing_id, model_version, raw_class, confidence, threshold) -> ProvenanceDecision:
    if not P.CLASSIFICATION_PROVENANCE_GUARD_ENABLED:
        return NOT_REJECTED
    # Hanya prediksi yang kelak menjadi alert (fault, keyakinan >= ambang) yang relevan.
    if raw_class not in FAULT_CLASSES or confidence < threshold:
        return NOT_REJECTED

    allowed = P.ALLOWED_BEARING_MODELS.get(model_version)
    bearing_model = None
    if allowed is None:
        reason = "model_version_not_in_allowlist"
    else:
        bearing_model = lookup_bearing_model(bearing_id)
        if bearing_model is None:
            reason = "bearing_model_unknown"
        elif bearing_model in allowed:
            return ProvenanceDecision(False, None, bearing_model)
        else:
            reason = "bearing_model_not_allowed"

    logger.warning(
        "provenance_guard: prediksi fault DITOLAK -> UNKNOWN reason=%s bearing_id=%s bearing_model=%r "
        "model_version=%s raw_class=%s confidence=%.3f",
        reason, bearing_id, bearing_model, model_version, raw_class, confidence,
    )
    return ProvenanceDecision(True, reason, bearing_model)
