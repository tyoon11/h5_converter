"""
라벨 맵핑 스크립트
==================
각 데이터셋의 원본 소스에서 진단 라벨을 추출합니다.

출력:
  1. label_definitions/{dataset}_labels.json  — 라벨 정의 (이름, 설명, 소스)
  2. label_definitions/{dataset}_labels.csv   — 라벨 테이블 (key + binary 컬럼)
  3. 기존 *_table.csv에서 라벨 컬럼 제거 (--clean_table)

지원 데이터셋:
  physionet  — old physionet2021_table.csv의 is_* 컬럼 (31개)
  heedb      — raw 12SL_diagnoses/diagnoses.csv + dictionary (149개)
  zzu        — raw AttributesDictionary.csv AHA/CHN/ICD-10 코드 (36개)
  code15     — 기존 CSV에서 추출 (6개)
  cpsc2021   — raw .hea 파일의 # comment 라벨 (3개)
  mimic4     — 라벨 소스 없음 (스킵)

실행:
  # 전체 데이터셋
  python append_labels.py --all

  # 특정 데이터셋만
  python append_labels.py --dataset physionet

  # 라벨 정의만 생성 (CSV 생성 없이)
  python append_labels.py --dataset heedb --dry_run

  # 기존 table CSV에서 라벨 컬럼도 제거
  python append_labels.py --all --clean_table
"""

import os
import sys
import glob
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ═══════════════════════════════════════════════════════════════
# 경로 상수
# ═══════════════════════════════════════════════════════════════
H5_ROOT = Path("/home/irteam/ddn-opendata1/h5")
RAW_ROOT = Path("/home/irteam/ddn-opendata1/raw")

DATASET_CONFIGS = {
    "physionet": {
        "h5_dir": H5_ROOT / "physionet/v2.0",
        "csv": H5_ROOT / "physionet/v2.0/ecg_table.csv",
        "file_name_csv": H5_ROOT / "physionet/v2.0/file_name.csv",
    },
    "heedb": {
        "h5_dir": H5_ROOT / "heedb/v4.0",
        "csv": H5_ROOT / "heedb/v4.0/heedb_table.csv",
    },
    "zzu": {
        "h5_dir": H5_ROOT / "ZZU-pECG/v2.0",
        "csv": H5_ROOT / "ZZU-pECG/v2.0/ecg_table.csv",
    },
    "cpsc2021": {
        "h5_dir": H5_ROOT / "cpsc2021/v2.0",
        "csv": H5_ROOT / "cpsc2021/v2.0/ecg_table.csv",
    },
    "code15": {
        "h5_dir": H5_ROOT / "code15/v2.0",
        "csv": H5_ROOT / "code15/v2.0/code15_table.csv",
    },
}


# ═══════════════════════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════════════════════
def _label_dir(dataset: str) -> Path:
    """라벨 파일을 저장할 디렉토리 (각 데이터셋 H5 폴더)"""
    return DATASET_CONFIGS[dataset]["h5_dir"]


def save_label_def(dataset: str, label_map: dict, description: str):
    """라벨 정의를 JSON으로 저장합니다."""
    out_dir = _label_dir(dataset)
    out = {
        "dataset": dataset,
        "description": description,
        "n_labels": len(label_map),
        "labels": label_map,
    }
    path = out_dir / f"{dataset}_labels.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logging.info(f"  라벨 정의 저장: {path} ({len(label_map)}개)")
    return path


BASE_TABLE_COLS = {
    "filepath", "dataset", "pid", "rid", "sid", "oid",
    "age", "gender", "height", "weight", "fs", "channel_name",
    "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
    "bs_corr", "bs_dtw",
}


def save_label_csv(dataset: str, csv_path: Path, rows_labels: dict,
                   label_cols: list, key_cols: list, key_fn=None):
    """
    라벨 CSV를 label_definitions/{dataset}_labels.csv 에 저장합니다.

    rows_labels: key → set of label column names that are True
    key_fn:      row → key (default: (pid, rid))
    key_cols:    라벨 CSV에 포함할 key 컬럼 (filepath, pid, rid, oid 등)
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # 라벨 컬럼 생성
    for col in label_cols:
        df[col] = False

    col_locs = {col: df.columns.get_loc(col) for col in label_cols}
    matched = 0

    for i, row in df.iterrows():
        if key_fn:
            key = key_fn(row)
        else:
            key = (str(row["pid"]), int(row["rid"]))
        labels = rows_labels.get(key, set())
        if labels:
            matched += 1
            for col in labels:
                if col in col_locs:
                    df.iat[i, col_locs[col]] = True
        if (i + 1) % 1_000_000 == 0:
            logging.info(f"    {i+1:,}/{len(df):,} matched={matched:,}")

    # 라벨 CSV 저장 (key + 라벨만)
    valid_keys = [c for c in key_cols if c in df.columns]
    label_df = df[valid_keys + label_cols]
    label_csv_path = _label_dir(dataset) / f"{dataset}_labels.csv"
    label_df.to_csv(label_csv_path, index=False)
    logging.info(f"  라벨 CSV: {label_csv_path.name} "
                 f"({len(label_df):,}행, key {len(valid_keys)} + 라벨 {len(label_cols)}, "
                 f"라벨 보유={matched:,}/{len(df):,} {100*matched/len(df):.1f}%)")
    return matched, len(df)


def clean_table_csv(csv_path: Path):
    """table CSV에서 라벨 컬럼을 제거합니다."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path, low_memory=False)
    keep = [c for c in df.columns if c in BASE_TABLE_COLS]
    removed = [c for c in df.columns if c not in BASE_TABLE_COLS]
    if not removed:
        return
    df[keep].to_csv(csv_path, index=False)
    logging.info(f"  정리: {csv_path.name} {len(keep)+len(removed)} → {len(keep)}컬럼 "
                 f"(라벨 {len(removed)}개 제거)")


# ═══════════════════════════════════════════════════════════════
# physionet — old CSV에서 is_* 컬럼 가져오기
# ═══════════════════════════════════════════════════════════════
def map_physionet(dry_run=False):
    logging.info("\n=== physionet 라벨 맵핑 ===")

    old_csv = H5_ROOT / "old/physionet2021_table.csv"
    if not old_csv.exists():
        logging.error(f"  old CSV 없음: {old_csv}")
        return

    old = pd.read_csv(old_csv, dtype=str, low_memory=False)
    base_cols = {"", "Unnamed: 0", "Unnamed: 0.1", "filepath", "pid", "rid", "sid",
                 "oid", "age", "gender", "height", "weight", "channel_name",
                 "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
                 "bs_corr", "bs_dtw"}
    label_cols = [c for c in old.columns if c not in base_cols]
    logging.info(f"  old CSV 라벨: {len(label_cols)}개")

    label_map = {col: col for col in label_cols}
    save_label_def("physionet", label_map,
                   "PhysioNet Challenge 2021 — SNOMED-based binary labels from old CSV")

    if dry_run:
        return

    # old pid → label dict
    label_dict = {}
    for _, row in old.iterrows():
        pid = str(row.get("pid", "")).strip()
        active = {col for col in label_cols
                  if str(row.get(col, "")).strip().lower() in ("true", "1", "1.0")}
        label_dict[pid] = active

    # file_name.csv로 new filepath → original_filename 매핑
    cfg = DATASET_CONFIGS["physionet"]
    fn_df = pd.read_csv(cfg["file_name_csv"])
    fn_map = dict(zip(fn_df["h5_filepath"], fn_df["original_filename"].astype(str)))

    # filepath → original_filename → label set 매핑 구축
    def _build_fp_labels(df):
        rows = {}
        for i, row in df.iterrows():
            orig = fn_map.get(str(row["filepath"]), "")
            rows[(str(row["pid"]), int(row["rid"]))] = label_dict.get(orig, set())
        return rows

    # 라벨 CSV 생성 (ecg_table.csv 기준)
    ecg_csv = cfg["csv"]
    df = pd.read_csv(ecg_csv, low_memory=False)
    rows_labels = _build_fp_labels(df)
    key_cols = ["filepath", "dataset", "pid", "rid", "oid"]
    save_label_csv("physionet", ecg_csv, rows_labels, label_cols, key_cols)


# ═══════════════════════════════════════════════════════════════
# heedb — 12SL diagnoses.csv + dictionary
# ═══════════════════════════════════════════════════════════════
HEEDB_LABEL_NAMES = [
    "NORMAL_SINUS_RHYTHM", "SINUS_RHYTHM", "SINUS_BRADYCARDIA", "SINUS_TACHYCARDIA",
    "MARKED_SINUS_BRADYCARDIA", "WITH_SINUS_ARRHYTHMIA", "WITH_MARKED_SINUS_ARRHYTHMIA",
    "WITH_SINUS_PAUSE", "ATRIAL_FIBRILLATION", "PREMATURE_ATRIAL_COMPLEXES",
    "ATRIAL_FLUTTER", "ECTOPIC_ATRIAL_RHYTHM", "WITH_RAPID_VENTRICULAR_RESPONSE",
    "WITH_SLOW_VENTRICULAR_RESPONSE", "PREMATURE_SUPRAVENTRICULAR_COMPLEXES",
    "SUPRAVENTRICULAR_TACHYCARDIA", "MULTIFOCAL_ATRIAL_TACHYCARDIA",
    "JUNCTIONAL_RHYTHM", "JUNCTIONAL_BRADYCARDIA", "WITH_RETROGRADE_CONDUCTION",
    "WITH_JUNCTIONAL_ESCAPE_COMPLEXES", "WITH_A_COMPETING_JUNCTIONAL_PACEMAKER",
    "SINUS_ATRIAL_CAPTURE", "NO_P_WAVES_FOUND", "BIATRIAL_ENLARGEMENT",
    "PREMATURE_VENTRICULAR_COMPLEXES", "FUSION_COMPLEXES",
    "PREMATURE_VENTRICULAR_AND_FUSION_COMPLEXES", "VENTRICULAR_TACHYCARDIA",
    "WIDE_QRS_TACHYCARDIA", "WIDE_QRS_RHYTHM", "IDIOVENTRICULAR_RHYTHM",
    "VENTRICULAR_PACED_COMPLEXES", "WITH_VENTRICULAR_ESCAPE_COMPLEXES",
    "IN_A_PATTERN_OF_BIGEMINY",
    "WITH_PREMATURE_VENTRICULAR_OR_ABERRANTLY_CONDUCTED_COMPLEXES",
    "ELECTRONIC_ATRIAL_PACEMAKER", "ELECTRONIC_VENTRICULAR_PACEMAKER",
    "ATRIAL_PACED_RHYTHM", "VENTRICULAR_PACED_RHYTHM",
    "ATRIAL_SENSED_VENTRICULAR_PACED_RHYTHM",
    "AV_SEQUENTIAL_OR_DUAL_CHAMBER_ELECTRONIC_PACEMAKER",
    "AV_DUAL_PACED_RHYTHM", "AV_DUAL_PACED_COMPLEXES", "ATRIAL_PACED_COMPLEXES",
    "BIVENTRICULAR_PACEMAKER_DETECTED", "ELECTRONIC_DEMAND_PACING",
    "SUSPECT_UNSPECIFIED_PACEMAKER_FAILURE",
    "WITH_1ST_DEGREE_AV_BLOCK", "WITH_PROLONGED_AV_CONDUCTION",
    "WITH_VARIABLE_AV_BLOCK", "WITH_2ND_DEGREE_SA_BLOCK_MOBITZ_I",
    "WITH_2ND_DEGREE_SA_BLOCK_MOBITZ_II", "WITH_2ND_DEGREE_AV_BLOCK_MOBITZ_I",
    "WITH_2_1_AV_CONDUCTION", "WITH_AV_DISSOCIATION", "WITH_COMPLETE_HEART_BLOCK",
    "RIGHT_BUNDLE_BRANCH_BLOCK", "LEFT_BUNDLE_BRANCH_BLOCK",
    "INCOMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK", "INCOMPLETE_LEFT_BUNDLE_BRANCH_BLOCK",
    "LEFT_ANTERIOR_FASCICULAR_BLOCK", "LEFT_POSTERIOR_FASCICULAR_BLOCK",
    "BIFASCICULAR_BLOCK", "RBBB_AND_LEFT_ANTERIOR_FASCICULAR_BLOCK",
    "RBBB_AND_LEFT_POSTERIOR_FASCICULAR_BLOCK", "MASKED_BY_FASCICULAR_BLOCK",
    "ABERRANT_CONDUCTION", "NONSPECIFIC_INTRAVENTRICULAR_CONDUCTION_DELAY",
    "NONSPECIFIC_INTRAVENTRICULAR_BLOCK", "WITH_QRS_WIDENING",
    "WITH_QRS_WIDENING_AND_REPOLARIZATION_ABNORMALITY",
    "WOLFF_PARKINSON_WHITE", "WITH_SHORT_PR", "BLOCKED",
    "RSR_OR_QR_PATTERN_IN_V1_SUGGESTS_RVCD", "RSR_PATTERN_IN_V1",
    "PULMONARY_DISEASE_PATTERN",
    "SEPTAL_INFARCT", "ANTERIOR_INFARCT", "LATERAL_INFARCT",
    "ANTEROSEPTAL_INFARCT", "INFERIOR_INFARCT", "ANTEROLATERAL_INFARCT",
    "ACUTE_MI_STEMI", "INFERIOR_POSTERIOR_INFARCT", "POSTERIOR_INFARCT",
    "ACUTE_MI", "ACUTE_PERICARDITIS",
    "CONSIDER_RV_INVOLVEMENT_IN_ACUTE_INFERIOR_INFARCT",
    "LATERAL_INJURY_PATTERN", "INFERIOR_INJURY_PATTERN",
    "ANTERIOR_INJURY_PATTERN", "INFEROLATERAL_INJURY_PATTERN",
    "ANTEROLATERAL_INJURY_PATTERN",
    "LEFT_VENTRICULAR_HYPERTROPHY", "VOLTAGE_CRITERIA_FOR_LVH",
    "RIGHT_VENTRICULAR_HYPERTROPHY",
    "LEFT_ATRIAL_ENLARGEMENT", "RIGHT_ATRIAL_ENLARGEMENT",
    "BIVENTRICULAR_HYPERTROPHY",
    "NONSPECIFIC_ST_ABNORMALITY", "NONSPECIFIC_ST_AND_T_WAVE_ABNORMALITY",
    "NONSPECIFIC_T_WAVE_ABNORMALITY",
    "NONSPECIFIC_T_WAVE_ABNORMALITY_NOW_EVIDENT_IN",
    "NONSPECIFIC_T_WAVE_ABNORMALITY_NO_LONGER_EVIDENT_IN",
    "T_WAVE_INVERSION_NOW_EVIDENT_IN", "T_WAVE_INVERSION_NO_LONGER_EVIDENT_IN",
    "T_WAVE_INVERSION_LESS_EVIDENT_IN", "T_WAVE_INVERSION_MORE_EVIDENT_IN",
    "T_WAVE_AMPLITUDE_HAS_DECREASED_IN", "T_WAVE_AMPLITUDE_HAS_INCREASED_IN",
    "INVERTED_T_REPLACED_NONSPECIFIC_T_IN",
    "NONSPECIFIC_T_REPLACED_INVERTED_T_IN",
    "NON_SPECIFIC_CHANGE_IN_ST_SEGMENT_IN",
    "ST_NOW_DEPRESSED_IN", "ST_NO_LONGER_DEPRESSED_IN",
    "ST_LESS_DEPRESSED_IN", "ST_MORE_DEPRESSED_IN",
    "ST_ELEVATION_NOW_PRESENT_IN", "ST_NO_LONGER_ELEVATED_IN",
    "ST_LESS_ELEVATED_IN", "ST_MORE_ELEVATED_IN",
    "ST_ELEVATION_HAS_REPLACED_ST_DEPRESSION_IN",
    "WITH_REPOLARIZATION_ABNORMALITY", "EARLY_REPOLARIZATION",
    "OR_DIGITALIS_EFFECT", "ACUTE",
    "LEFT_AXIS_DEVIATION", "RIGHT_AXIS_DEVIATION", "RIGHTWARD_AXIS",
    "RIGHT_SUPERIOR_AXIS_DEVIATION", "LEFTWARD_AXIS",
    "ABNORMAL_LEFT_AXIS_DEVIATION", "ABNORMAL_RIGHT_AXIS_DEVIATION",
    "R_IN_AVL", "LOW_VOLTAGE_QRS",
    "QT_HAS_SHORTENED", "QT_HAS_LENGTHENED", "PROLONGED_QT",
    "WITH_UNDETERMINED_RHYTHM_IRREGULARITY", "UNDETERMINED_RHYTHM",
    "ANTEROLATERAL_LEADS", "SUPRAVENTRICULAR_COMPLEXES",
    "PEDIATRIC_ECG_ANALYSIS",
    "ABNORMAL_ECG", "NORMAL_ECG", "OTHERWISE_NORMAL_ECG", "BORDERLINE_ECG",
]


def _build_heedb_code_to_label():
    """사전에서 code → label column name 매핑을 구축합니다."""
    dict_df = pd.read_csv(RAW_ROOT / "heedb/ECG/I0001/12SL_diagnoses/diagnoses_dictionary.csv")
    desc_lower_to_code = {}
    for _, r in dict_df.iterrows():
        desc_lower_to_code[str(r["diagnoses"]).strip().lower()] = str(r["codes"]).strip()

    # 149개 라벨명 → description(lower) 매핑
    LABEL_TO_DESC = {
        "NORMAL_SINUS_RHYTHM": "normal sinus rhythm",
        "SINUS_RHYTHM": "sinus rhythm",
        "SINUS_BRADYCARDIA": "sinus bradycardia",
        "SINUS_TACHYCARDIA": "sinus tachycardia",
        "MARKED_SINUS_BRADYCARDIA": "marked sinus bradycardia",
        "WITH_SINUS_ARRHYTHMIA": "with sinus arrhythmia",
        "WITH_MARKED_SINUS_ARRHYTHMIA": "with marked sinus arrhythmia",
        "WITH_SINUS_PAUSE": "with sinus pause",
        "ATRIAL_FIBRILLATION": "atrial fibrillation",
        "PREMATURE_ATRIAL_COMPLEXES": "premature atrial complexes",
        "ATRIAL_FLUTTER": "atrial flutter",
        "ECTOPIC_ATRIAL_RHYTHM": "ectopic atrial rhythm",
        "WITH_RAPID_VENTRICULAR_RESPONSE": "with rapid ventricular response",
        "WITH_SLOW_VENTRICULAR_RESPONSE": "with slow ventricular response",
        "PREMATURE_SUPRAVENTRICULAR_COMPLEXES": "premature supraventricular complexes",
        "SUPRAVENTRICULAR_TACHYCARDIA": "supraventricular tachycardia",
        "MULTIFOCAL_ATRIAL_TACHYCARDIA": "multifocal atrial tachycardia",
        "JUNCTIONAL_RHYTHM": "junctional rhythm",
        "JUNCTIONAL_BRADYCARDIA": "junctional bradycardia",
        "WITH_RETROGRADE_CONDUCTION": "with retrograde conduction",
        "WITH_JUNCTIONAL_ESCAPE_COMPLEXES": "with junctional escape complexes",
        "WITH_A_COMPETING_JUNCTIONAL_PACEMAKER": "with a competing junctional pacemaker",
        "SINUS_ATRIAL_CAPTURE": "sinus/atrial capture",
        "NO_P_WAVES_FOUND": "(no p-waves found)",
        "BIATRIAL_ENLARGEMENT": "biatrial enlargement",
        "PREMATURE_VENTRICULAR_COMPLEXES": "premature ventricular complexes",
        "FUSION_COMPLEXES": "fusion complexes",
        "PREMATURE_VENTRICULAR_AND_FUSION_COMPLEXES": "premature ventricular and fusion complexes",
        "VENTRICULAR_TACHYCARDIA": "ventricular tachycardia (ventricular or supraventricular with aberration)",
        "WIDE_QRS_TACHYCARDIA": "wide qrs tachycardia",
        "WIDE_QRS_RHYTHM": "wide qrs rhythm",
        "IDIOVENTRICULAR_RHYTHM": "idioventricular rhythm with av block",
        "VENTRICULAR_PACED_COMPLEXES": "ventricular-paced complexes",
        "WITH_VENTRICULAR_ESCAPE_COMPLEXES": "with ventricular escape complexes",
        "IN_A_PATTERN_OF_BIGEMINY": "in a pattern of bigeminy",
        "WITH_PREMATURE_VENTRICULAR_OR_ABERRANTLY_CONDUCTED_COMPLEXES": "with premature ventricular or aberrantly conducted complexes",
        "ELECTRONIC_ATRIAL_PACEMAKER": "electronic atrial pacemaker",
        "ELECTRONIC_VENTRICULAR_PACEMAKER": "electronic ventricular pacemaker",
        "ATRIAL_PACED_RHYTHM": "atrial-paced rhythm",
        "VENTRICULAR_PACED_RHYTHM": "ventricular-paced rhythm",
        "ATRIAL_SENSED_VENTRICULAR_PACED_RHYTHM": "atrial-sensed ventricular-paced rhythm",
        "AV_SEQUENTIAL_OR_DUAL_CHAMBER_ELECTRONIC_PACEMAKER": "av sequential or dual chamber electronic pacemaker",
        "AV_DUAL_PACED_RHYTHM": "av dual-paced rhythm",
        "AV_DUAL_PACED_COMPLEXES": "av dual-paced complexes",
        "ATRIAL_PACED_COMPLEXES": "atrial-paced complexes",
        "BIVENTRICULAR_PACEMAKER_DETECTED": "biventricular pacemaker detected",
        "ELECTRONIC_DEMAND_PACING": "electronic demand pacing",
        "SUSPECT_UNSPECIFIED_PACEMAKER_FAILURE": "*** suspect unspecified pacemaker failure",
        "WITH_1ST_DEGREE_AV_BLOCK": "with 1st degree av block",
        "WITH_PROLONGED_AV_CONDUCTION": "with prolonged av conduction",
        "WITH_VARIABLE_AV_BLOCK": "with variable av block",
        "WITH_2ND_DEGREE_SA_BLOCK_MOBITZ_I": "with 2nd degree sa block (mobitz i)",
        "WITH_2ND_DEGREE_SA_BLOCK_MOBITZ_II": "with 2nd degree sa block (mobitz ii)",
        "WITH_2ND_DEGREE_AV_BLOCK_MOBITZ_I": "with 2nd degree av block (mobitz i)",
        "WITH_2_1_AV_CONDUCTION": "with 2:1 av conduction",
        "WITH_AV_DISSOCIATION": "with av dissociation",
        "WITH_COMPLETE_HEART_BLOCK": "with complete heart block",
        "RIGHT_BUNDLE_BRANCH_BLOCK": "right bundle branch block",
        "LEFT_BUNDLE_BRANCH_BLOCK": "left bundle branch block",
        "INCOMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK": "incomplete right bundle branch block",
        "INCOMPLETE_LEFT_BUNDLE_BRANCH_BLOCK": "incomplete left bundle branch block",
        "LEFT_ANTERIOR_FASCICULAR_BLOCK": "left anterior fascicular block",
        "LEFT_POSTERIOR_FASCICULAR_BLOCK": "left posterior fascicular block",
        "BIFASCICULAR_BLOCK": "*** bifascicular block ***",
        "RBBB_AND_LEFT_ANTERIOR_FASCICULAR_BLOCK": "(rbbb and left anterior fascicular block)",
        "RBBB_AND_LEFT_POSTERIOR_FASCICULAR_BLOCK": "(rbbb and left posterior fascicular block)",
        "MASKED_BY_FASCICULAR_BLOCK": "(masked by fascicular block?)",
        "ABERRANT_CONDUCTION": "aberrant conduction",
        "NONSPECIFIC_INTRAVENTRICULAR_CONDUCTION_DELAY": "nonspecific intraventricular conduction delay",
        "NONSPECIFIC_INTRAVENTRICULAR_BLOCK": "nonspecific intraventricular block",
        "WITH_QRS_WIDENING": "with qrs widening",
        "WITH_QRS_WIDENING_AND_REPOLARIZATION_ABNORMALITY": "with qrs widening and repolarization abnormality",
        "WOLFF_PARKINSON_WHITE": "wolff-parkinson-white",
        "WITH_SHORT_PR": "with short pr",
        "BLOCKED": "blocked",
        "RSR_OR_QR_PATTERN_IN_V1_SUGGESTS_RVCD": "rsr' or qr pattern in v1 suggests right ventricular conduction delay",
        "RSR_PATTERN_IN_V1": "rsr' pattern in v1",
        "PULMONARY_DISEASE_PATTERN": "pulmonary disease pattern",
        "SEPTAL_INFARCT": "septal infarct",
        "ANTERIOR_INFARCT": "anterior infarct",
        "LATERAL_INFARCT": "lateral infarct",
        "ANTEROSEPTAL_INFARCT": "anteroseptal infarct",
        "INFERIOR_INFARCT": "inferior infarct",
        "ANTEROLATERAL_INFARCT": "anterolateral infarct",
        "ACUTE_MI_STEMI": "** ** acute mi / stemi ** **",
        "INFERIOR_POSTERIOR_INFARCT": "inferior-posterior infarct",
        "POSTERIOR_INFARCT": "posterior infarct",
        "ACUTE_MI": "** ** acute mi ** **",
        "ACUTE_PERICARDITIS": "acute pericarditis",
        "CONSIDER_RV_INVOLVEMENT_IN_ACUTE_INFERIOR_INFARCT": "consider right ventricular involvement in acute inferior infarct",
        "LATERAL_INJURY_PATTERN": "lateral injury pattern",
        "INFERIOR_INJURY_PATTERN": "inferior injury pattern",
        "ANTERIOR_INJURY_PATTERN": "anterior injury pattern",
        "INFEROLATERAL_INJURY_PATTERN": "inferolateral injury pattern",
        "ANTEROLATERAL_INJURY_PATTERN": "anterolateral injury pattern",
        "LEFT_VENTRICULAR_HYPERTROPHY": "left ventricular hypertrophy",
        "VOLTAGE_CRITERIA_FOR_LVH": "voltage criteria for left ventricular hypertrophy",
        "RIGHT_VENTRICULAR_HYPERTROPHY": "right ventricular hypertrophy",
        "LEFT_ATRIAL_ENLARGEMENT": "left atrial enlargement",
        "RIGHT_ATRIAL_ENLARGEMENT": "right atrial enlargement",
        "BIVENTRICULAR_HYPERTROPHY": "biventricular hypertrophy",
        "NONSPECIFIC_ST_ABNORMALITY": "nonspecific st abnormality",
        "NONSPECIFIC_ST_AND_T_WAVE_ABNORMALITY": "nonspecific st and t wave abnormality",
        "NONSPECIFIC_T_WAVE_ABNORMALITY": "nonspecific t wave abnormality",
        "NONSPECIFIC_T_WAVE_ABNORMALITY_NOW_EVIDENT_IN": "nonspecific t wave abnormality now evident in",
        "NONSPECIFIC_T_WAVE_ABNORMALITY_NO_LONGER_EVIDENT_IN": "nonspecific t wave abnormality no longer evident in",
        "T_WAVE_INVERSION_NOW_EVIDENT_IN": "t wave inversion now evident in",
        "T_WAVE_INVERSION_NO_LONGER_EVIDENT_IN": "t wave inversion no longer evident in",
        "T_WAVE_INVERSION_LESS_EVIDENT_IN": "t wave inversion less evident in",
        "T_WAVE_INVERSION_MORE_EVIDENT_IN": "t wave inversion more evident in",
        "T_WAVE_AMPLITUDE_HAS_DECREASED_IN": "t wave amplitude has decreased in",
        "T_WAVE_AMPLITUDE_HAS_INCREASED_IN": "t wave amplitude has increased in",
        "INVERTED_T_REPLACED_NONSPECIFIC_T_IN": "inverted t waves have replaced nonspecific t wave abnormality in",
        "NONSPECIFIC_T_REPLACED_INVERTED_T_IN": "nonspecific t wave abnormality has replaced inverted t waves in",
        "NON_SPECIFIC_CHANGE_IN_ST_SEGMENT_IN": "non-specific change in st segment in",
        "ST_NOW_DEPRESSED_IN": "st now depressed in",
        "ST_NO_LONGER_DEPRESSED_IN": "st no longer depressed in",
        "ST_LESS_DEPRESSED_IN": "st less depressed in",
        "ST_MORE_DEPRESSED_IN": "st more depressed in",
        "ST_ELEVATION_NOW_PRESENT_IN": "st elevation now present in",
        "ST_NO_LONGER_ELEVATED_IN": "st no longer elevated in",
        "ST_LESS_ELEVATED_IN": "st less elevated in",
        "ST_MORE_ELEVATED_IN": "st more elevated in",
        "ST_ELEVATION_HAS_REPLACED_ST_DEPRESSION_IN": "st elevation has replaced st depression in",
        "WITH_REPOLARIZATION_ABNORMALITY": "with repolarization abnormality",
        "EARLY_REPOLARIZATION": "early repolarization",
        "OR_DIGITALIS_EFFECT": "or digitalis effect",
        "ACUTE": "acute",
        "LEFT_AXIS_DEVIATION": "left axis deviation",
        "RIGHT_AXIS_DEVIATION": "right axis deviation",
        "RIGHTWARD_AXIS": "rightward axis",
        "RIGHT_SUPERIOR_AXIS_DEVIATION": "right superior axis deviation",
        "LEFTWARD_AXIS": "leftward axis",
        "ABNORMAL_LEFT_AXIS_DEVIATION": "abnormal left axis deviation",
        "ABNORMAL_RIGHT_AXIS_DEVIATION": "abnormal right axis deviation",
        "R_IN_AVL": "r in avl",
        "LOW_VOLTAGE_QRS": "low voltage qrs",
        "QT_HAS_SHORTENED": "qt has shortened",
        "QT_HAS_LENGTHENED": "qt has lengthened",
        "PROLONGED_QT": "prolonged qt",
        "WITH_UNDETERMINED_RHYTHM_IRREGULARITY": "with undetermined rhythm irregularity",
        "UNDETERMINED_RHYTHM": "undetermined rhythm",
        "ANTEROLATERAL_LEADS": "anterolateral leads",
        "SUPRAVENTRICULAR_COMPLEXES": "supraventricular complexes",
        "PEDIATRIC_ECG_ANALYSIS": "** * pediatric ecg analysis * **",
        "ABNORMAL_ECG": "abnormal ecg",
        "NORMAL_ECG": "normal ecg",
        "OTHERWISE_NORMAL_ECG": "otherwise normal ecg",
        "BORDERLINE_ECG": "borderline ecg",
    }

    code_to_label = {}
    for label, desc in LABEL_TO_DESC.items():
        code = desc_lower_to_code.get(desc)
        if code:
            code_to_label[code] = label

    return code_to_label, LABEL_TO_DESC


def map_heedb(dry_run=False):
    logging.info("\n=== heedb 라벨 맵핑 ===")
    from convert_to_h5 import normalize_pid

    code_to_label, label_to_desc = _build_heedb_code_to_label()
    label_map = {label: desc for label, desc in label_to_desc.items() if any(
        v == label for v in code_to_label.values())}
    save_label_def("heedb", {l: d for l, d in label_to_desc.items()},
                   "HEEDB 12SL GE algorithm — 149 diagnostic statements")

    if dry_run:
        return

    # diagnoses.csv 로드
    def load_diag(inst):
        path = RAW_ROOT / f"heedb/ECG/{inst}/12SL_diagnoses/diagnoses.csv"
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df["_fn"] = (df["FileName"].str.strip().str.strip('"')
                     .str.replace("\n", "", regex=False).str.lstrip("./")
                     .str.replace(".hea", "", regex=False))
        df["_codes"] = df["codes"].apply(
            lambda x: set(c.strip() for c in str(x).strip().strip('"').split(",") if c.strip())
            if pd.notna(x) else set()
        )
        return dict(zip(df["_fn"], df["_codes"]))

    logging.info("  diagnoses 로드...")
    d1 = load_diag("I0001"); d6 = load_diag("I0006")
    logging.info(f"    I0001: {len(d1):,}, I0006: {len(d6):,}")

    # metadata → (pid, rid) → code set
    label_dict = {}
    for inst, diag_map in [("I0001", d1), ("I0006", d6)]:
        meta = pd.read_csv(RAW_ROOT / f"heedb/ECG/{inst}/metadata/metadata.csv",
                           dtype=str, low_memory=False, usecols=["BDSPPatientID", "FileName"])
        for rid, row in meta.iterrows():
            pid = normalize_pid(row["BDSPPatientID"])
            fn = str(row["FileName"]).strip().lstrip("/")
            if fn.startswith("WFDB/"): fn = fn[5:]
            fn = fn.replace(".hea", "")
            codes = diag_map.get(fn, set())
            active_labels = {code_to_label[c] for c in codes if c in code_to_label}
            label_dict[(pid, rid)] = active_labels
        logging.info(f"    {inst}: {len(meta):,}행")
    del d1, d6

    label_cols = list(code_to_label.values())
    key_cols = ["filepath", "pid", "rid", "oid"]
    save_label_csv("heedb", DATASET_CONFIGS["heedb"]["csv"],
                   label_dict, label_cols, key_cols)


# ═══════════════════════════════════════════════════════════════
# zzu — AttributesDictionary.csv AHA/CHN codes
# ═══════════════════════════════════════════════════════════════
def map_zzu(dry_run=False):
    logging.info("\n=== zzu 라벨 맵핑 ===")

    ecg_code = pd.read_csv(RAW_ROOT / "ZZU-pECG/ECGCode.csv")
    attr = pd.read_csv(RAW_ROOT / "ZZU-pECG/AttributesDictionary.csv")

    # AHA/CHN code → description 매핑
    aha_to_desc = {}
    for _, row in ecg_code.iterrows():
        desc = str(row["Description"]).strip()
        for col in ["AHA(Category&Code)", "CHN(Category&Code)"]:
            val = str(row.get(col, "")).strip()
            if val and val != "nan" and not val.startswith("N(") and not val.startswith("J("):
                for v in val.split("+"):
                    v = v.strip()
                    if not v.startswith("Modifier"):
                        aha_to_desc[v] = desc

    MODIFIER_MAP = {
        "Modifier362": "Depression", "Modifier363": "Elevation",
        "Modifier367": "Inversion", "Modifier308": "Occasional",
        "Modifier310": "Frequent",
    }

    def parse_codes(aha_val, chn_val):
        diags = set()
        for val in [aha_val, chn_val]:
            if pd.isna(val): continue
            for part in str(val).split(";"):
                part = part.strip().strip("'")
                if not part: continue
                if "+" in part:
                    parts = part.split("+")
                    base = aha_to_desc.get(parts[0].strip(), parts[0].strip())
                    mod = MODIFIER_MAP.get(parts[1].strip(), parts[1].strip())
                    diags.add(f"{base}_{mod}")
                elif part in aha_to_desc:
                    diags.add(aha_to_desc[part])
                else:
                    diags.add(part)
        return diags

    # 전체 진단 추출 + 빈도
    attr["_diags"] = attr.apply(lambda r: parse_codes(r.get("AHA_code"), r.get("CHN_code")), axis=1)
    diag_freq = Counter()
    for d in attr["_diags"]:
        for dx in d:
            diag_freq[dx] += 1

    MIN_COUNT = 100
    CLEAN_LABELS = {
        "Normal ECG": "is_Normal", "Otherwise normal ECG": "is_Normal_Other",
        "Sinus tachycardia": "is_STach", "Sinus bradycardia": "is_SBrady",
        "Sinus arrhythmia": "is_SArr", "T-wave abnormality": "is_TAb",
        "ST deviation with T-wave change": "is_ST_T", "ST deviation": "is_ST",
        "ST deviation_Depression": "is_STDep", "ST deviation_Elevation": "is_STElev",
        "Right ventricular hypertrophy": "is_RVH", "Left ventricular hypertrophy": "is_LVH",
        "Left ventricular high voltage": "is_LVHV",
        "Right bundle-branch block": "is_RBBB",
        "Incomplete right bundle-branch block": "is_IRBBB",
        "Left anterior fascicular block": "is_LAnFB",
        "Abnormal Q wave": "is_QAb", "Prolonged QT interval": "is_LQT",
        "Prolonged QTc interval": "is_LQTc", "Prominent U waves": "is_UWave",
        "Right-axis deviation": "is_RAD", "Low voltage": "is_LowV",
        "Right atrial enlargement": "is_RAE", "Left atrial enlargement": "is_LAE",
        "Early repolarization": "is_EarlyRepol",
        "Hyperkalemia": "is_Hyperkalemia", "Hypocalcemia": "is_Hypocalcemia",
        "Junctional tachycardia": "is_JTach",
        "Junctional escape complex(es)": "is_JEsc",
        "Ectopic atrial tachycardia, unifocal": "is_EATach",
        "First-degree AV block": "is_IAVB",
        "AV block, complete (third-degree)": "is_CAVB",
        "Ventricular preexcitation": "is_WPW",
        "AV dissociation": "is_AVDiss",
        "Atrial premature complex(es)": "is_APC",
        "Ventricular premature complex(es)": "is_VPC",
        "Atrial escape complex(es)": "is_AEsc",
        "Hypokalemia or drug effect": "is_Hypokalemia",
    }
    selected = {diag: col for diag, col in CLEAN_LABELS.items()
                if diag_freq.get(diag, 0) >= MIN_COUNT}

    save_label_def("zzu", {col: diag for diag, col in selected.items()},
                   f"ZZU-pECG — AHA/CHN code based, ≥{MIN_COUNT} occurrences")

    if dry_run:
        return

    # (pid, rid) → label set
    label_dict = {}
    for idx, row in attr.iterrows():
        pid = str(row["Patient_ID"])
        rid = idx
        diags = row["_diags"]
        active = {selected[d] for d in diags if d in selected}
        label_dict[(pid, rid)] = active

    label_cols = list(selected.values())
    key_cols = ["filepath", "dataset", "pid", "rid", "oid"]
    save_label_csv("zzu", DATASET_CONFIGS["zzu"]["csv"],
                   label_dict, label_cols, key_cols)


# ═══════════════════════════════════════════════════════════════
# cpsc2021 — .hea comments
# ═══════════════════════════════════════════════════════════════
def map_cpsc2021(dry_run=False):
    logging.info("\n=== cpsc2021 라벨 맵핑 ===")

    LABEL_MAP = {
        "non atrial fibrillation":        "is_nonAF",
        "persistent atrial fibrillation": "is_persistentAF",
        "paroxysmal atrial fibrillation": "is_paroxysmalAF",
    }
    save_label_def("cpsc2021", {v: k for k, v in LABEL_MAP.items()},
                   "CPSC2021 AF detection — 3 classes from .hea comments")

    if dry_run:
        return

    import wfdb
    cfg = DATASET_CONFIGS["cpsc2021"]
    fn_df = pd.read_csv(H5_ROOT / "cpsc2021/v2.0/file_name.csv")

    # original_filepath → label → (pid, rid) 기반 dict
    fp_to_orig = dict(zip(fn_df["h5_filepath"], fn_df["original_filepath"]))
    ecg_df = pd.read_csv(cfg["csv"], low_memory=False)

    label_dict = {}
    for i, row in ecg_df.iterrows():
        orig = fp_to_orig.get(str(row["filepath"]), "")
        labels = set()
        try:
            rec = wfdb.rdrecord(orig)
            for c in (getattr(rec, "comments", []) or []):
                cl = c.strip().lstrip("#").strip().lower()
                if cl in LABEL_MAP:
                    labels.add(LABEL_MAP[cl])
        except Exception:
            pass
        label_dict[(str(row["pid"]), int(row["rid"]))] = labels

    label_cols = list(LABEL_MAP.values())
    key_cols = ["filepath", "dataset", "pid", "rid", "oid"]
    save_label_csv("cpsc2021", cfg["csv"], label_dict, label_cols, key_cols)


# ═══════════════════════════════════════════════════════════════
# code15 — 기존 CSV에서 라벨 추출
# ═══════════════════════════════════════════════════════════════
def map_code15(dry_run=False):
    logging.info("\n=== code15 라벨 맵핑 ===")

    CODE15_LABELS = ["1dAVb", "RBBB", "LBBB", "is_SB", "is_STach", "is_AF"]
    label_map = {col: col for col in CODE15_LABELS}
    save_label_def("code15", label_map,
                   "CODE-15% — 6 binary rhythm/conduction labels")

    if dry_run:
        return

    cfg = DATASET_CONFIGS["code15"]
    csv_path = cfg["csv"]
    if not csv_path.exists():
        logging.error(f"  CSV 없음: {csv_path}")
        return

    # 현재 CSV 또는 old CSV에서 라벨 가져오기
    df = pd.read_csv(csv_path, low_memory=False)
    existing_labels = [c for c in CODE15_LABELS if c in df.columns]

    if not existing_labels:
        # table에서 제거됐을 수 있음 → old CSV에서 가져오기
        old_csv = H5_ROOT / "old/code15_table.csv"
        if not old_csv.exists():
            logging.warning("  code15 라벨 소스 없음 (현재 CSV 및 old CSV)")
            return
        logging.info(f"  old CSV에서 라벨 추출: {old_csv}")
        old = pd.read_csv(old_csv, dtype=str, low_memory=False)
        existing_labels = [c for c in CODE15_LABELS if c in old.columns]
        if not existing_labels:
            logging.warning("  old CSV에도 라벨 없음")
            return
        # old pid+rid → label
        label_dict = {}
        for _, row in old.iterrows():
            pid = str(row.get("pid", "")).strip()
            rid_val = str(row.get("rid", "")).strip()
            active = {col for col in existing_labels
                      if str(row.get(col, "")).strip().lower() in ("true", "1", "1.0")}
            label_dict[(pid, rid_val)] = active

        key_cols = ["filepath", "pid", "rid", "oid"]
        save_label_csv("code15", csv_path, label_dict, existing_labels, key_cols,
                       key_fn=lambda row: (str(row["pid"]), str(int(row["rid"]))))
        return

    key_cols = ["filepath", "pid", "rid", "oid"]
    valid_keys = [c for c in key_cols if c in df.columns]
    label_df = df[valid_keys + existing_labels]
    label_csv_path = _label_dir("code15") / "code15_labels.csv"
    label_df.to_csv(label_csv_path, index=False)
    logging.info(f"  라벨 CSV: {label_csv_path.name} "
                 f"({len(label_df):,}행, key {len(valid_keys)} + 라벨 {len(existing_labels)})")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="ECG 데이터셋별 라벨 맵핑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python append_labels.py --all
  python append_labels.py --dataset physionet
  python append_labels.py --dataset heedb
  python append_labels.py --dataset cpsc2021
  python append_labels.py --dataset heedb --dry_run   # 라벨 정의만 생성
        """,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="전체 데이터셋 라벨 맵핑")
    target.add_argument("--dataset", type=str,
                        choices=["physionet", "heedb", "zzu", "cpsc2021", "code15"],
                        help="특정 데이터셋만")
    parser.add_argument("--dry_run", action="store_true",
                        help="라벨 정의 JSON만 생성 (라벨 CSV 생성 없이)")
    parser.add_argument("--clean_table", action="store_true",
                        help="기존 table CSV에서 라벨 컬럼 제거")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    MAPPERS = {
        "physionet": map_physionet,
        "heedb":     map_heedb,
        "zzu":       map_zzu,
        "cpsc2021":  map_cpsc2021,
        "code15":    map_code15,
    }

    if args.all:
        targets = ["physionet", "zzu", "cpsc2021", "code15", "heedb"]
    else:
        targets = [args.dataset]

    for ds in targets:
        MAPPERS[ds](dry_run=args.dry_run)

    # table CSV에서 라벨 컬럼 제거
    if args.clean_table and not args.dry_run:
        logging.info("\n=== table CSV 라벨 컬럼 제거 ===")
        for ds in targets:
            cfg = DATASET_CONFIGS[ds]
            clean_table_csv(cfg["csv"])

    logging.info("\n완료!")
    for ds in targets:
        d = _label_dir(ds)
        logging.info(f"  {ds}: {d}/{ds}_labels.json, {ds}_labels.csv")


if __name__ == "__main__":
    main()
