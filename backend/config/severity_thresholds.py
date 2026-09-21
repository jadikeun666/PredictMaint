"""
Default severity thresholds + alert cooldown.

Sumber nilai: scheduling.md §1 (severity) dan prd.md D1 (cooldown 24 jam).
Nilai default FIXED oleh spec; mengubahnya di deployment adalah perubahan
konfigurasi, bukan perubahan arsitektur. Satu-satunya tempat angka ini
didefinisikan — jangan hardcode ulang di kode lain (engineering-rules.md).
"""

# --- RUL-based severity (jam) ---
RUL_CRITICAL_MAX_HOURS = 168   # <= 7 hari  -> CRITICAL
RUL_HIGH_MAX_HOURS = 336       # <= 14 hari -> HIGH
RUL_WARNING_MAX_HOURS = 720    # <= 30 hari -> WARNING ; > 720 -> INFO (tanpa alert)

# --- Classification-based severity (confidence) ---
CLASSIFICATION_CRITICAL_MIN_CONFIDENCE = 0.85   # >= 0.85        -> CRITICAL
CLASSIFICATION_HIGH_MIN_CONFIDENCE = 0.6        # 0.6 <= c < 0.85 -> HIGH

# --- Cooldown alert (prd.md D1) ---
ALERT_COOLDOWN_HOURS = 24
