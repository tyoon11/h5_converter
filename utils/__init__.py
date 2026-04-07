"""
utils 패키지
============
ECG H5 변환 파이프라인의 공유 유틸리티 모듈입니다.

Modules:
  h5_structure       : H5 파일 구조 생성 (create_h5_structure, TARGET_SIG_NAME)
  signal_processing  : 신호 처리 유틸리티 (reorder, statistics, beat, fiducial)
"""

from .h5_structure import (
    create_h5_structure,
    TARGET_SIG_NAME,
    FIDUCIAL_FEATURE_KEYS,
    UTF8,
)
from .signal_processing import (
    reorder_signal,
    has_zero_lead,
    signal_statistics,
    beat_similarity,
    extract_beat_annotation,
    extract_fiducial,
)

__all__ = [
    "create_h5_structure",
    "TARGET_SIG_NAME",
    "FIDUCIAL_FEATURE_KEYS",
    "UTF8",
    "reorder_signal",
    "has_zero_lead",
    "signal_statistics",
    "beat_similarity",
    "extract_beat_annotation",
    "extract_fiducial",
]
