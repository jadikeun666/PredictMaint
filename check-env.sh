#!/usr/bin/env bash
# ============================================================
# PredictMaint - Environment & Prerequisite Check Script
# Jalankan di WSL2: bash check-env.sh
# Murni diagnostik, tidak mengubah/menginstall apa pun.
# ============================================================

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}OK${NC}   - $1"; }
miss() { echo -e "  ${RED}MISS${NC} - $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC} - $1"; }

section() {
  echo ""
  echo -e "${BOLD}=== $1 ===${NC}"
}

check_cmd() {
  # $1 = command name, $2 = version flag, $3 = human label
  if command -v "$1" >/dev/null 2>&1; then
    VER=$($1 $2 2>&1 | head -n1)
    ok "$3 terpasang -> $VER"
  else
    miss "$3 TIDAK ditemukan (command '$1' not found)"
  fi
}

echo -e "${BOLD}PredictMaint — Environment Check${NC}"
echo "Waktu: $(date)"
echo "Host: $(hostname)"
echo "WSL kernel: $(uname -r)"

# ------------------------------------------------------------
section "1. Docker & Docker Compose"
# ------------------------------------------------------------
check_cmd docker "--version" "Docker"
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "Docker daemon berjalan (docker info sukses)"
  else
    miss "Docker daemon TIDAK berjalan / tidak bisa diakses (cek Docker Desktop / WSL integration)"
  fi
fi

if docker compose version >/dev/null 2>&1; then
  ok "Docker Compose (plugin v2) -> $(docker compose version | head -n1)"
elif command -v docker-compose >/dev/null 2>&1; then
  warn "Hanya ditemukan 'docker-compose' (v1 lama) -> $(docker-compose --version)"
else
  miss "Docker Compose TIDAK ditemukan (baik plugin v2 maupun docker-compose v1)"
fi

# ------------------------------------------------------------
section "2. Python"
# ------------------------------------------------------------
check_cmd python3 "--version" "Python3"
if command -v pip3 >/dev/null 2>&1; then
  ok "pip3 terpasang -> $(pip3 --version)"
  # test apakah --break-system-packages diperlukan/dikenali
  PIP_HELP=$(pip3 install --help 2>&1)
  if echo "$PIP_HELP" | grep -q "break-system-packages"; then
    ok "pip3 mendukung flag --break-system-packages (PEP 668 externally-managed env terdeteksi)"
  else
    warn "flag --break-system-packages tidak dikenali pip3 versi ini (mungkin tidak perlu, cek manual)"
  fi
else
  miss "pip3 TIDAK ditemukan"
fi

if command -v python3 >/dev/null 2>&1; then
  if python3 -m venv --help >/dev/null 2>&1; then
    ok "Modul venv tersedia (python3 -m venv)"
  else
    miss "Modul venv TIDAK tersedia (install: apt install python3-venv)"
  fi
fi

# ------------------------------------------------------------
section "3. Node.js & Frontend Tooling"
# ------------------------------------------------------------
check_cmd node "--version" "Node.js"
check_cmd npm "--version" "npm"
if command -v yarn >/dev/null 2>&1; then
  ok "yarn terpasang -> $(yarn --version)"
else
  warn "yarn tidak ditemukan (opsional, npm cukup untuk Vite)"
fi

# ------------------------------------------------------------
section "4. Service Lokal (Postgres/Redis/InfluxDB/Mosquitto)"
# ------------------------------------------------------------
echo "  (Diasumsikan semua akan jalan via Docker Compose. Cek ini hanya untuk"
echo "   mendeteksi instalasi lokal yang mungkin bentrok port dengan container.)"

check_local_service() {
  local name="$1" cmd="$2" port="$3"
  if command -v "$cmd" >/dev/null 2>&1; then
    warn "$name binary TERPASANG lokal ($cmd ditemukan) - pastikan tidak konflik dengan container"
  else
    ok "$name tidak terpasang lokal (akan jalan via Docker Compose - sesuai rencana)"
  fi
  if command -v ss >/dev/null 2>&1; then
    if ss -tuln 2>/dev/null | grep -q ":$port "; then
      warn "Port $port ($name) SUDAH DIPAKAI oleh proses lain di mesin ini"
    else
      ok "Port $port ($name) kosong/tersedia"
    fi
  fi
}

check_local_service "PostgreSQL" "psql" "5432"
check_local_service "Redis"      "redis-cli" "6379"
check_local_service "InfluxDB"   "influx" "8086"
check_local_service "Mosquitto"  "mosquitto" "1883"

# ------------------------------------------------------------
section "5. Dataset Check (sesuaikan DATASET_ROOT di bawah)"
# ------------------------------------------------------------
# UBAH path ini sesuai lokasi dataset Anda:
DATASET_ROOT="${DATASET_ROOT:-$HOME/predictmaint-datasets}"

echo "  Mencari dataset di: $DATASET_ROOT"
echo "  (Set env var DATASET_ROOT=/path/lain sebelum menjalankan script jika berbeda)"

if [ -d "$DATASET_ROOT" ]; then
  ok "Folder dataset root ditemukan: $DATASET_ROOT"
  echo ""
  echo "  --- Struktur 3 level pertama ---"
  find "$DATASET_ROOT" -maxdepth 3 2>/dev/null | sed "s|$DATASET_ROOT|.|"

  echo ""
  echo "  --- Deteksi dataset spesifik ---"
  CWRU_FOUND=$(find "$DATASET_ROOT" -maxdepth 4 -iname "*cwru*" 2>/dev/null)
  PU_FOUND=$(find "$DATASET_ROOT" -maxdepth 4 \( -iname "*paderborn*" -o -iname "K00[1-6]" -o -iname "KA0*" -o -iname "KI0*" \) 2>/dev/null)
  IMS_FOUND=$(find "$DATASET_ROOT" -maxdepth 4 -iname "*IMS*" -o -iname "*1st_test*" -o -iname "*2nd_test*" -o -iname "*3rd_test*" 2>/dev/null)
  CMAPSS_FOUND=$(find "$DATASET_ROOT" -maxdepth 4 -iname "*C-MAPSS*" -o -iname "*CMAPSS*" -o -iname "*Damage*Propagation*" 2>/dev/null)

  [ -n "$CWRU_FOUND" ] && ok "CWRU terdeteksi:" && echo "$CWRU_FOUND" | sed 's/^/       /' || miss "CWRU TIDAK terdeteksi"
  [ -n "$PU_FOUND" ] && ok "Paderborn (PU) terdeteksi:" && echo "$PU_FOUND" | sed 's/^/       /' || miss "Paderborn (PU) TIDAK terdeteksi"
  [ -n "$IMS_FOUND" ] && ok "IMS Bearing Data terdeteksi:" && echo "$IMS_FOUND" | sed 's/^/       /' || miss "IMS Bearing Data TIDAK terdeteksi"
  [ -n "$CMAPSS_FOUND" ] && ok "NASA C-MAPSS terdeteksi:" && echo "$CMAPSS_FOUND" | sed 's/^/       /' || warn "NASA C-MAPSS tidak terdeteksi (bonus/reference only, tidak wajib)"
else
  miss "Folder dataset root TIDAK ditemukan di $DATASET_ROOT"
  echo "  -> Jalankan ulang dengan: DATASET_ROOT=/path/asli/dataset bash check-env.sh"
fi

# ------------------------------------------------------------
section "6. File Referensi (Mosquitto source & buku DSP)"
# ------------------------------------------------------------
REF_SEARCH_PATHS=("$HOME" "$HOME/Downloads" "$(pwd)")
MOSQ_FOUND=""
DSP_FOUND=""
for p in "${REF_SEARCH_PATHS[@]}"; do
  [ -d "$p" ] || continue
  F1=$(find "$p" -maxdepth 3 -iname "*mosquitto*2*1*2*" 2>/dev/null | head -n5)
  F2=$(find "$p" -maxdepth 3 -iname "*Scientist*Engineer*DSP*" -o -iname "*guide*digital*signal*" 2>/dev/null | head -n5)
  [ -n "$F1" ] && MOSQ_FOUND="$MOSQ_FOUND
$F1"
  [ -n "$F2" ] && DSP_FOUND="$DSP_FOUND
$F2"
done

if [ -n "$MOSQ_FOUND" ]; then
  ok "Mosquitto source terdeteksi:"
  echo "$MOSQ_FOUND" | sed '/^$/d;s/^/       /'
else
  miss "mosquitto-2_1_2_tar.gz TIDAK ditemukan di \$HOME/\$HOME/Downloads/cwd"
fi

if [ -n "$DSP_FOUND" ]; then
  ok "Buku DSP (Scientist and Engineer's Guide) terdeteksi:"
  echo "$DSP_FOUND" | sed '/^$/d;s/^/       /'
else
  miss "The_Scientist_and_Engineer_Guide_to_DSP.zip TIDAK ditemukan di lokasi umum"
fi

# ------------------------------------------------------------
section "7. Disk Space & Resource"
# ------------------------------------------------------------
echo "  --- Disk usage (root filesystem) ---"
df -h / 2>/dev/null | sed 's/^/  /'
echo ""
echo "  --- Memory ---"
free -h 2>/dev/null | sed 's/^/  /'

# ------------------------------------------------------------
section "8. Git (opsional, untuk versioning project)"
# ------------------------------------------------------------
check_cmd git "--version" "Git"

echo ""
echo -e "${BOLD}=== Selesai. Copy seluruh output di atas dan paste ke chat Claude. ===${NC}"