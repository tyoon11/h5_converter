#!/bin/bash
# =============================================================
# ECG H5 전체 변환 파이프라인
# 실행: bash run_convert.sh [GROUP]
#   GROUP: heedb | physionet | zzu | all (기본: all)
# =============================================================

set -e  # 오류 시 중단

# ─────────────────────────────────────────────────────────────
# 경로 설정 (환경에 맞게 수정하세요)
# ─────────────────────────────────────────────────────────────
HEEDB_RAW=/home/irteam/opendata1/raw/heedb/ECG
PHYSIONET_RAW=/home/irteam/ddn-opendata1/raw/physionet.org/files
ZZU_RAW=/home/irteam/ddn-opendata1/raw/ZZU-pECG

OUTPUT_ROOT=/home/irteam/ddn-opendata1/h5/all/v1.0
NUM_CPUS=64

GROUP=${1:-all}   # 인수 없으면 all

SCRIPT_DIR=$(dirname "$(realpath "$0")")
CSV="$OUTPUT_ROOT/ecg_table.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────
# 1. H5 변환
# ─────────────────────────────────────────────────────────────
log "===== [1/3] H5 변환 시작 (group=$GROUP) ====="
python "$SCRIPT_DIR/convert_to_h5.py" \
    --group          "$GROUP" \
    --heedb_root     "$HEEDB_RAW" \
    --physionet_root "$PHYSIONET_RAW" \
    --zzu_root       "$ZZU_RAW" \
    --output_root    "$OUTPUT_ROOT" \
    --num_cpus       $NUM_CPUS
log "===== [1/3] H5 변환 완료 ====="

# ─────────────────────────────────────────────────────────────
# 2. Fiducial 추가 (beat_annotation + fiducial_point/feature)
# ─────────────────────────────────────────────────────────────
log "===== [2/3] Fiducial 추가 시작 ====="
python "$SCRIPT_DIR/append_fiducial.py" \
    --csv      "$CSV" \
    --h5_root  "$OUTPUT_ROOT" \
    --num_cpus $NUM_CPUS
log "===== [2/3] Fiducial 추가 완료 ====="

# ─────────────────────────────────────────────────────────────
# 3. Signal quality 추가 (bs_corr, bs_dtw, 신호 통계)
# ─────────────────────────────────────────────────────────────
log "===== [3/3] Signal quality 추가 시작 ====="
python "$SCRIPT_DIR/append_signal_quality.py" \
    --csv      "$CSV" \
    --h5_root  "$OUTPUT_ROOT" \
    --num_cpus $NUM_CPUS
log "===== [3/3] Signal quality 추가 완료 ====="

log "============================================="
log "전체 파이프라인 완료"
log "  출력: $OUTPUT_ROOT"
log "  CSV:  $CSV"
log "============================================="
