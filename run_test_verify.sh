#!/bin/bash
# =============================================================
# ECG H5 변환 테스트 + 검증 파이프라인
# 실행: bash run_test_verify.sh [GROUP]
#   GROUP: heedb | physionet | zzu | all (기본: all)
#
# 단계:
#   1. test_convert.py  — 데이터셋별 1개 레코드 변환 + CSV 생성/컬럼 검증
#   2. verify_h5.py     — 데이터셋별 1개 파일 상세 inspect + 일괄 검증
# =============================================================

set -e  # 오류 시 중단

# ─────────────────────────────────────────────────────────────
# 경로 설정 (환경에 맞게 수정하세요)
# ─────────────────────────────────────────────────────────────
HEEDB_RAW=/home/irteam/ddn-opendata1/raw/heedb/ECG
PHYSIONET_RAW=/home/irteam/ddn-opendata1/raw/physionet.org/files
ZZU_RAW=/home/irteam/ddn-opendata1/raw/ZZU-pECG

OUTPUT_ROOT=/home/irteam/local-node-d/tykim/convert_h5/convert_raw_to_h5/tmp/h5_test_verify
SAMPLE_N=50      # verify_h5 일괄 검증 시 데이터셋별 샘플 수 (0 또는 빈 값이면 전체)

GROUP=${1:-all}  # 인수 없으면 all

SCRIPT_DIR=$(dirname "$(realpath "$0")")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$OUTPUT_ROOT"

# ─────────────────────────────────────────────────────────────
# 1. test_convert.py — 데이터셋별 1개 변환 + CSV 컬럼 검증
# ─────────────────────────────────────────────────────────────
log "===== [1/2] test_convert.py 시작 (group=$GROUP) ====="
python "$SCRIPT_DIR/test_convert.py" \
    --group          "$GROUP" \
    --heedb_root     "$HEEDB_RAW" \
    --physionet_root "$PHYSIONET_RAW" \
    --zzu_root       "$ZZU_RAW" \
    --output_root    "$OUTPUT_ROOT" \
    --compute_beat \
    --compute_fiducial
log "===== [1/2] test_convert.py 완료 ====="

# ─────────────────────────────────────────────────────────────
# 2. verify_h5.py — 데이터셋별 1개 상세 + 일괄 검증
# ─────────────────────────────────────────────────────────────
log "===== [2/2] verify_h5.py 시작 ====="
if [ -n "$SAMPLE_N" ] && [ "$SAMPLE_N" -gt 0 ]; then
    python "$SCRIPT_DIR/verify_h5.py" \
        --output_root "$OUTPUT_ROOT" \
        --sample      "$SAMPLE_N"
else
    python "$SCRIPT_DIR/verify_h5.py" \
        --output_root "$OUTPUT_ROOT"
fi
log "===== [2/2] verify_h5.py 완료 ====="

log "============================================="
log "테스트 + 검증 완료"
log "  출력 루트       : $OUTPUT_ROOT"
log "  H5 데이터       : $OUTPUT_ROOT/data/"
log "  테스트 CSV      : $OUTPUT_ROOT/ecg_table_test.csv"
log "============================================="
