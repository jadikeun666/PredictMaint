"""
Severity computation — scheduling.md §1. Fungsi murni (tanpa akses DB).

- Classification: HEALTHY/UNKNOWN tidak pernah menghasilkan alert.
- RUL: > 720 jam = INFO = tidak ada alert.
- Kalau input klasifikasi DAN RUL sama-sama ada, severity yang LEBIH TINGGI
  yang menentukan (CRITICAL > HIGH > WARNING > INFO). Saat seri, klasifikasi
  menang (FAULT_DETECTED).
"""
from dataclasses import dataclass
from typing import Optional

from apps.alerting.models import Alert
from config import severity_thresholds as T

_RANK = {
    Alert.Severity.INFO: 0,
    Alert.Severity.WARNING: 1,
    Alert.Severity.HIGH: 2,
    Alert.Severity.CRITICAL: 3,
}

FAULT_CLASSES = frozenset({"INNER_RACE", "OUTER_RACE", "BALL"})


@dataclass(frozen=True)
class SeverityResult:
    severity: str
    alert_type: str


def classification_severity(fault_class, confidence) -> Optional[str]:
    if fault_class not in FAULT_CLASSES or confidence is None:
        return None
    if confidence >= T.CLASSIFICATION_CRITICAL_MIN_CONFIDENCE:
        return Alert.Severity.CRITICAL
    if confidence >= T.CLASSIFICATION_HIGH_MIN_CONFIDENCE:
        return Alert.Severity.HIGH
    return None


def rul_severity(rul_hours) -> Optional[str]:
    if rul_hours is None:
        return None
    if rul_hours <= T.RUL_CRITICAL_MAX_HOURS:
        return Alert.Severity.CRITICAL
    if rul_hours <= T.RUL_HIGH_MAX_HOURS:
        return Alert.Severity.HIGH
    if rul_hours <= T.RUL_WARNING_MAX_HOURS:
        return Alert.Severity.WARNING
    return Alert.Severity.INFO


def compute_severity(fault_class=None, confidence=None, rul_hours=None) -> Optional[SeverityResult]:
    """Return SeverityResult, atau None kalau tidak ada yang layak di-alert."""
    candidates = []
    c = classification_severity(fault_class, confidence)
    if c is not None:
        candidates.append(SeverityResult(c, Alert.AlertType.FAULT_DETECTED))
    r = rul_severity(rul_hours)
    if r is not None:
        candidates.append(SeverityResult(r, Alert.AlertType.RUL_THRESHOLD))
    candidates = [x for x in candidates if x.severity != Alert.Severity.INFO]
    if not candidates:
        return None
    # max() mengembalikan elemen pertama saat seri -> klasifikasi menang.
    return max(candidates, key=lambda x: _RANK[x.severity])
