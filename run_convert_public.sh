#!/bin/bash
# =============================================================
# Public ECG 전체 변환 파이프라인
# 실행: bash run_convert_public.sh
# =============================================================

set -e  # 오류 시 중단

PHYSIONET_RAW=/home/irteam/ddn-opendata1/raw/physionet.org/files
ZZU_RAW=/home/irteam/ddn-opendata1/raw/ZZU-pECG

PHYSIONET_OUT=/home/irteam/ddn-opendata1/h5/physionet/v2.0
ZZU_OUT=/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0

NUM_CPUS=64
SCRIPT_DIR=$(dirname "$(realpath "$0")")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# =============================================================
# 1. physionet H5 변환
# =============================================================
log "===== [1/6] physionet H5 변환 시작 ====="
python "$SCRIPT_DIR/convert_to_h5_public.py" \
    --group physionet \
    --physionet_root "$PHYSIONET_RAW" \
    --output_root    "$PHYSIONET_OUT" \
    --num_cpus       $NUM_CPUS
log "===== [1/6] physionet H5 변환 완료 ====="

# =============================================================
# 2. physionet fiducial 추가
# =============================================================
log "===== [2/6] physionet fiducial 추가 시작 ====="
python "$SCRIPT_DIR/append_fiducial.py" \
    --csv     "$PHYSIONET_OUT/public_ecg_table.csv" \
    --h5_root "$PHYSIONET_OUT" \
    --num_cpus $NUM_CPUS
log "===== [2/6] physionet fiducial 추가 완료 ====="

# =============================================================
# 3. physionet signal quality 추가
# =============================================================
log "===== [3/6] physionet signal quality 추가 시작 ====="
python "$SCRIPT_DIR/append_signal_quality.py" \
    --csv     "$PHYSIONET_OUT/public_ecg_table.csv" \
    --h5_root "$PHYSIONET_OUT" \
    --num_cpus $NUM_CPUS
log "===== [3/6] physionet signal quality 추가 완료 ====="

# =============================================================
# 4. ZZU H5 변환
# =============================================================
log "===== [4/6] ZZU H5 변환 시작 ====="
python "$SCRIPT_DIR/convert_to_h5_public.py" \
    --group   zzu \
    --zzu_root "$ZZU_RAW" \
    --output_root "$ZZU_OUT" \
    --num_cpus $NUM_CPUS
log "===== [4/6] ZZU H5 변환 완료 ====="

# =============================================================
# 5. ZZU fiducial 추가
# =============================================================
log "===== [5/6] ZZU fiducial 추가 시작 ====="
python "$SCRIPT_DIR/append_fiducial.py" \
    --csv     "$ZZU_OUT/public_ecg_table.csv" \
    --h5_root "$ZZU_OUT" \
    --num_cpus $NUM_CPUS
log "===== [5/6] ZZU fiducial 추가 완료 ====="

# =============================================================
# 6. ZZU signal quality 추가
# =============================================================
log "===== [6/6] ZZU signal quality 추가 시작 ====="
python "$SCRIPT_DIR/append_signal_quality.py" \
    --csv     "$ZZU_OUT/public_ecg_table.csv" \
    --h5_root "$ZZU_OUT" \
    --num_cpus $NUM_CPUS
log "===== [6/6] ZZU signal quality 추가 완료 ====="

log "============================================="
log "전체 파이프라인 완료"
log "  physionet: $PHYSIONET_OUT"
log "  zzu:       $ZZU_OUT"
log "============================================="