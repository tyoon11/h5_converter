"""
기존 H5 → 새 표준 H5 변환 스크립트
=====================================
code15 / mimic4 / physionet2021 의 구 포맷 H5를 HEEDB 표준 포맷으로 변환합니다.

[구 포맷]
  ecg/ecg_metadata/  (attrs: fs, signal_len, channel_num, ...)
  ecg/segments/0/signal/{I, II, ...}   ← lead별 개별 dataset
  ecg/segments/0/signal_quality/       ← nan_ratio, amp_mean, bs_correlation, ...
  ecg/segments/0/fiducial_feature/     ← attrs
  ecg/segments/0/fiducial_point/       ← fsample(uint16), fiducial(object)
  patient_info/                        ← attrs: age, gender, ...
  root_attribute/                      ← attrs: file_name, dataset_version, ...

[새 포맷 (HEEDB 표준)]
  root attrs: dataset_version, file_name, beat_ext_method, fidu_extract_method
  ECG/metadata/  (attrs: record_name, n_sig, fs, sig_len, base_time, base_date, dtype)
  ECG/segments/0/signal  shape=(12, T) float16  ← TARGET_SIG_NAME 순서로 stacked
  ECG/segments/0/fiducial_point/  fsample(int16), fiducial(UTF8)
  ECG/segments/0/fiducial_feature/  attrs
  ※ signal_quality, patient_info → CSV 컬럼으로만 보존

[파일명 / oid 규칙]
  file_name = {prefix}{pid}{rid}          ← 모두 소문자 prefix
  oid       = {prefix}{pid}{rid}{sid}
  code15       : cod
  mimic4       : m4
  physionet2021: 파일명에서 sub-prefix 추출 후 소문자화
    chapman_shaoxing      → psh
    cpsc_2018             → pcp
    cpsc_2018_extra       → pce
    georgia               → pge
    ningbo                → pnb
    ptb                   → ppt
    ptb-xl                → ppx
    st_petersburg_incart  → pin

실행:
  python convert_old_h5_to_new.py --dataset code15
  python convert_old_h5_to_new.py --dataset mimic4
  python convert_old_h5_to_new.py --dataset physionet2021
  python convert_old_h5_to_new.py --dataset all --num_cpus 32
  python convert_old_h5_to_new.py --dataset code15 --dry_run
"""

import os
import sys
import ast
import argparse
import logging
import numpy as np
import pandas as pd
import h5py
import ray
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heedb.create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME, FIDUCIAL_FEATURE_KEYS

# ─────────────────────────────────────────────────────────────
# prefix 규칙 (모두 소문자)
# ─────────────────────────────────────────────────────────────
# 단일 prefix 데이터셋
DATASET_PREFIX = {
    "code15": "cod",
    "mimic4": "m4",
}

# physionet2021 sub-dataset prefix (파일명 앞부분 매칭용, 소문자 기준)
# 구 파일명에 이미 대소문자 섞인 prefix가 포함되어 있으므로 lower() 비교
PHYSIONET_PREFIX_MAP = {
    "psh": "psh",   # chapman_shaoxing
    "pcp": "pcp",   # cpsc_2018
    "pce": "pce",   # cpsc_2018_extra
    "pge": "pge",   # georgia
    "pnb": "pnb",   # ningbo
    "ppt": "ppt",   # ptb
    "ppx": "ppx",   # ptb-xl
    "pin": "pin",   # st_petersburg_incart
}
PHYSIONET_KNOWN_PREFIXES = sorted(PHYSIONET_PREFIX_MAP.keys(), key=len, reverse=True)  # 긴 것 우선 매칭


def extract_physionet_prefix(old_filename: str) -> str:
    """
    구 physionet2021 파일명에서 sub-dataset prefix를 추출합니다.
    파일명 앞부분(소문자)을 알려진 prefix 목록과 매칭합니다.
    예) 'PgeA00010.h5' → 'pge'
    """
    stem = os.path.splitext(os.path.basename(old_filename))[0].lower()
    for pfx in PHYSIONET_KNOWN_PREFIXES:
        if stem.startswith(pfx):
            return pfx
    return "p21"  # fallback


def get_prefix(dataset_name: str, old_filepath: str = "") -> str:
    """dataset_name 과 구 filepath 로부터 새 prefix(소문자) 결정"""
    if dataset_name in DATASET_PREFIX:
        return DATASET_PREFIX[dataset_name]
    if dataset_name == "physionet2021":
        return extract_physionet_prefix(old_filepath)
    return ""


# ─────────────────────────────────────────────────────────────
# 구 포맷 channel order 관련
# ─────────────────────────────────────────────────────────────
# 구 포맷: I,II,III,aVR,aVF,aVL,V1,V2,V3,V4,V5,V6
# 새 포맷: I,II,III,V1,V2,V3,V4,V5,V6,aVF,aVL,aVR
# signal 은 lead 이름으로 직접 접근하므로 reorder 불필요.
# quality 배열(12-elem)은 아래 인덱스로 재정렬.
OLD_TO_TARGET_IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11, 4, 5, 3]

QUALITY_COLS = ["nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis", "bs_corr", "bs_dtw"]

# ─────────────────────────────────────────────────────────────
# 데이터셋 설정
# ─────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "code15": {
        "old_csv":     "/home/irteam/ddn-opendata1/h5/old/code15_table.csv",
        "old_h5_dir":  "/home/irteam/ddn-opendata1/h5/old/code15",
        "output_root": "/home/irteam/ddn-opendata1/h5/code15/v2.0",
        "label_cols":  ["1dAVb", "RBBB", "LBBB", "is_SB", "is_STach", "is_AF"],
    },
    "mimic4": {
        "old_csv":     "/home/irteam/ddn-opendata1/h5/old/mimic4_table.csv",
        "old_h5_dir":  "/home/irteam/ddn-opendata1/h5/old/mimic4",
        "output_root": "/home/irteam/ddn-opendata1/h5/mimic4/v2.0",
        "label_cols":  [],
    },
    "physionet2021": {
        "old_csv":     "/home/irteam/ddn-opendata1/h5/old/physionet2021_table.csv",
        "old_h5_dir":  "/home/irteam/ddn-opendata1/h5/old/physionet2021",
        "output_root": "/home/irteam/ddn-opendata1/h5/physionet2021/v2.0",
        "label_cols":  [
            "is_AF", "is_AFL", "is_BBB", "is_Brady", "is_CLBBB", "is_CRBBB",
            "is_IAVB", "is_IRBBB", "is_LAD", "is_LAnFB", "is_LBBB", "is_LPR",
            "is_LQRSV", "is_LQT", "is_NSIVCB", "is_NSR", "is_PAC", "is_PR",
            "is_PRWP", "is_PVC", "is_QAb", "is_RAD", "is_RBBB", "is_SA",
            "is_SB", "is_STach", "is_SVPB", "is_TAb", "is_TInv", "is_VPB", "is_Death",
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
def get_reorder_idx(old_channels):
    """구 channel 순서 → TARGET_SIG_NAME 순서 인덱스"""
    old_lower = {ch.lower(): i for i, ch in enumerate(old_channels)}
    idx = []
    for name in TARGET_SIG_NAME:
        key = name.lower()
        if key not in old_lower:
            raise ValueError(f"채널 '{name}' 없음: {old_channels}")
        idx.append(old_lower[key])
    return idx


def reorder_quality_array(arr, reorder_idx):
    """12-element quality 배열을 TARGET 순서로 재정렬"""
    if not arr or len(arr) == 0:
        return [np.nan] * 12
    arr = list(arr)
    if len(arr) != 12:
        return arr
    return [arr[i] for i in reorder_idx]


def parse_str_list(val):
    """CSV에 '[...]' 문자열로 저장된 list 파싱"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return [np.nan] * 12
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [np.nan] * 12


# ─────────────────────────────────────────────────────────────
# 구 H5 읽기
# ─────────────────────────────────────────────────────────────
def read_old_h5(old_h5_path):
    try:
        with h5py.File(old_h5_path, "r") as f:
            # root_attribute
            ra = f.get("root_attribute", {})
            dataset_version = str(ra.attrs.get("dataset_version", "1.0"))

            # ecg_metadata
            meta = f["ecg"]["ecg_metadata"]
            fs       = int(meta.attrs.get("fs", 500))
            sig_len  = int(meta.attrs.get("signal_len", meta.attrs.get("sig_len", 0)))
            record_name = str(meta.attrs.get("record_name", ""))
            base_time   = str(meta.attrs.get("base_time", ""))
            if base_time in ("None", "none"):
                base_time = ""

            ch_name_raw = meta["channel_name"][()]
            old_channel_order = [
                (c.decode() if isinstance(c, bytes) else c) for c in ch_name_raw
            ]

            # signal: lead 이름으로 직접 TARGET 순서로 stack
            sig_grp = f["ecg"]["segments"]["0"]["signal"]
            leads = []
            for lead in TARGET_SIG_NAME:
                if lead in sig_grp:
                    leads.append(sig_grp[lead][()].astype(np.float16))
                else:
                    T = sig_len if sig_len > 0 else (leads[0].shape[0] if leads else 0)
                    leads.append(np.full(T, np.nan, dtype=np.float16))
            signal = np.stack(leads, axis=0)  # (12, T)

            # fiducial_point
            seg0 = f["ecg"]["segments"]["0"]
            fidu_extract_method = ""
            fp_out = {"fsample": [], "fiducial": []}
            if "fiducial_point" in seg0:
                fp_grp = seg0["fiducial_point"]
                fidu_extract_method = fp_grp.attrs.get("extraction_method", "neurokit2-dwt")
                if "fsample" in fp_grp and len(fp_grp["fsample"]) > 0:
                    fp_out["fsample"] = [int(x) for x in fp_grp["fsample"][()]]
                    fp_out["fiducial"] = [
                        (x.decode() if isinstance(x, bytes) else str(x))
                        for x in fp_grp["fiducial"][()]
                    ]

            # fiducial_feature
            ff_out = {k: np.float16(np.nan) for k in FIDUCIAL_FEATURE_KEYS}
            if "fiducial_feature" in seg0:
                for k in FIDUCIAL_FEATURE_KEYS:
                    v = seg0["fiducial_feature"].attrs.get(k, np.nan)
                    try:
                        ff_out[k] = np.float16(v)
                    except Exception:
                        ff_out[k] = np.float16(np.nan)

            # signal_quality (H5 내 원본 순서; fallback용)
            old_quality = {col: [np.nan] * 12 for col in QUALITY_COLS}
            if "signal_quality" in seg0:
                sq = seg0["signal_quality"]
                for new_key, old_key in [
                    ("nan_ratio",    "nan_ratio"),
                    ("amp_mean",     "amp_mean"),
                    ("amp_std",      "amp_std"),
                    ("amp_skewness", "amp_skewness"),
                    ("amp_kurtosis", "amp_kurtosis"),
                    ("bs_corr",      "bs_correlation"),
                    ("bs_dtw",       "bs_dtw"),
                ]:
                    if old_key in sq and len(sq[old_key]) == 12:
                        old_quality[new_key] = list(sq[old_key][()])

        return {
            "dataset_version":     dataset_version,
            "fidu_extract_method": fidu_extract_method,
            "record_name":         record_name,
            "fs":                  fs,
            "sig_len":             sig_len if sig_len > 0 else signal.shape[1],
            "base_time":           base_time,
            "signal":              signal,
            "old_channel_order":   old_channel_order,
            "fiducial_point":      fp_out,
            "fiducial_feature":    ff_out,
            "old_quality":         old_quality,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Ray remote: 1 레코드 변환
# ─────────────────────────────────────────────────────────────
@ray.remote
def convert_one(row_dict, old_h5_dir, new_h5_dir, label_cols, dataset_name, dry_run=False):
    # ── 구 H5 경로 결정
    old_filepath = str(row_dict.get("filepath", "")).strip()
    if os.path.isabs(old_filepath) and os.path.exists(old_filepath):
        old_h5_path = old_filepath
    else:
        old_h5_path = os.path.join(old_h5_dir, os.path.basename(old_filepath))

    if not os.path.exists(old_h5_path):
        return None

    # ── 구 H5 읽기
    data = read_old_h5(old_h5_path)
    if data is None:
        return None

    # ── 새 file_name / oid 생성
    pid    = str(row_dict.get("pid", ""))
    rid    = str(row_dict.get("rid", ""))
    sid    = row_dict.get("sid", 0)
    prefix = get_prefix(dataset_name, old_filepath)   # 소문자 보장

    file_name = f"{prefix}{pid}{rid}"          # {prefix}{pid}{rid}
    new_oid   = f"{prefix}{pid}{rid}{sid}"     # {prefix}{pid}{rid}{sid}
    new_h5_path = os.path.join(new_h5_dir, f"{file_name}.h5")

    # ── beat_annotation: fiducial_point의 ECG_R_Peaks 추출, 없으면 빈 배열
    fp = data["fiducial_point"]
    r_samples = [
        s for s, lbl in zip(fp["fsample"], fp["fiducial"])
        if lbl == "ECG_R_Peaks"
    ]
    n_beats = len(r_samples)
    beat_annotation = [{
        "sample":   r_samples,
        "symbol":   [""] * n_beats,
        "subtype":  [0]  * n_beats,
        "chan":      [0]  * n_beats,
        "num":       [0]  * n_beats,
        "aux_note": [""] * n_beats,
    }]

    # ── WFDB 메타 필드: 구 포맷에 없으므로 빈 배열로 채움
    n12 = 12
    empty_str12  = [""] * n12
    empty_f16_12 = [np.float16(0.0)] * n12
    empty_i16_12 = [0] * n12

    # ── 새 H5 작성
    if not dry_run:
        try:
            os.makedirs(new_h5_dir, exist_ok=True)
            with h5py.File(new_h5_path, "w") as h5f:
                create_h5_structure(
                    h5f,
                    file_name=file_name,
                    dataset_version=data["dataset_version"],
                    beat_ext_method="neurokit2",
                    fidu_extract_method=data["fidu_extract_method"],
                    record_name=data["record_name"],
                    n_sig=12,
                    fs=data["fs"],
                    sig_len=data["sig_len"],
                    base_time=data["base_time"],
                    base_date="",
                    sig_name=TARGET_SIG_NAME,
                    fmt=empty_str12,
                    adc_gain=empty_f16_12,
                    baseline=empty_i16_12,
                    units=empty_str12,
                    adc_res=empty_i16_12,
                    adc_zero=empty_i16_12,
                    signal=[data["signal"]],
                    seg_len=1,
                    beat_annotation=beat_annotation,
                    fiducial_point=[data["fiducial_point"]],
                    fiducial_feature=[data["fiducial_feature"]],
                )
        except Exception:
            return None

    # ── signal quality: CSV 우선 → fallback H5, 구 channel 순서 → TARGET 재정렬
    try:
        reorder_idx = get_reorder_idx(data["old_channel_order"])
    except ValueError:
        reorder_idx = OLD_TO_TARGET_IDX

    def get_quality(col_key):
        csv_val = row_dict.get(col_key, None)
        if csv_val is not None and not (isinstance(csv_val, float) and np.isnan(csv_val)):
            arr = parse_str_list(csv_val)
            if len(arr) == 12:
                return reorder_quality_array(arr, reorder_idx)
        return reorder_quality_array(data["old_quality"].get(col_key, [np.nan] * 12), reorder_idx)

    quality = {col: str(get_quality(col)) for col in QUALITY_COLS}

    # ── CSV row
    csv_row = {
        "filepath":     f"data/{file_name}.h5",
        "pid":          pid,
        "rid":          rid,
        "sid":          sid,
        "oid":          new_oid,
        "age":          row_dict.get("age", -1),
        "gender":       row_dict.get("gender", 0),
        "height":       row_dict.get("height", np.nan),
        "weight":       row_dict.get("weight", np.nan),
        "fs":           data["fs"],
        "channel_name": str(TARGET_SIG_NAME),
    }
    csv_row.update(quality)
    for col in label_cols:
        csv_row[col] = row_dict.get(col, np.nan)

    return csv_row


# ─────────────────────────────────────────────────────────────
# 테스트: 1건 변환 후 H5 구조 출력
# ─────────────────────────────────────────────────────────────
def verify_h5(h5_path):
    """변환된 H5 구조를 터미널에 출력"""
    print(f"\n{'='*60}")
    print(f"  {h5_path}")
    print(f"{'='*60}")
    with h5py.File(h5_path, "r") as f:
        print("\n[root attrs]")
        for k in ["dataset_version", "file_name", "beat_ext_method", "fidu_extract_method"]:
            print(f"  {k}: {f.attrs.get(k, '(missing)')}")

        meta = f["ECG/metadata"]
        print("\n[ECG/metadata attrs]")
        for k in ["record_name", "n_sig", "fs", "sig_len", "base_time", "base_date", "dtype"]:
            print(f"  {k}: {meta.attrs.get(k, '(missing)')}")
        print("[ECG/metadata datasets]")
        for k in ["sig_name", "fmt", "adc_gain", "baseline", "units", "adc_res", "adc_zero"]:
            if k in meta:
                ds_obj = meta[k]
                print(f"  {k}: shape={ds_obj.shape}, dtype={ds_obj.dtype}, val={ds_obj[()]}")
            else:
                print(f"  {k}: (missing)")

        segs = f["ECG/segments"]
        print(f"\n[ECG/segments] seg_len={segs.attrs.get('seg_len', '?')}")
        seg0 = segs["0"]

        if "signal" in seg0:
            sig = seg0["signal"][()]
            print(f"  signal: shape={sig.shape}, dtype={sig.dtype}")
            print(f"    min={np.nanmin(sig):.4f}, max={np.nanmax(sig):.4f}, "
                  f"nan={int(np.sum(np.isnan(sig)))}")

        if "beat_annotation" in seg0:
            ba = seg0["beat_annotation"]
            print(f"  beat_annotation: {ba['sample'].shape[0]}개 R-peak")
            print(f"    sample[:5] = {ba['sample'][:5]}")
        else:
            print("  beat_annotation: (missing)")

        if "fiducial_point" in seg0:
            fp = seg0["fiducial_point"]
            print(f"  fiducial_point: {fp['fsample'].shape[0]}개 포인트")
        else:
            print("  fiducial_point: (missing)")

        if "fiducial_feature" in seg0:
            ff = seg0["fiducial_feature"]
            nan_cnt = sum(
                1 for k in ff.attrs
                if isinstance(ff.attrs[k], (float, np.floating)) and np.isnan(ff.attrs[k])
            )
            print(f"  fiducial_feature: NaN {nan_cnt}/{len(ff.attrs)}개")
        else:
            print("  fiducial_feature: (missing)")

    # 필수 필드 검증
    issues = []
    with h5py.File(h5_path, "r") as f:
        for k in ["dataset_version", "file_name", "beat_ext_method", "fidu_extract_method"]:
            if k not in f.attrs:
                issues.append(f"root attr missing: {k}")
        meta = f.get("ECG/metadata")
        if meta is None:
            issues.append("missing ECG/metadata")
        else:
            for k in ["record_name", "n_sig", "fs", "sig_len"]:
                if k not in meta.attrs:
                    issues.append(f"metadata attr missing: {k}")
            for k in ["sig_name", "fmt", "adc_gain", "baseline", "units", "adc_res", "adc_zero"]:
                if k not in meta:
                    issues.append(f"metadata dataset missing: {k}")
        seg0 = f.get("ECG/segments/0")
        if seg0 is None:
            issues.append("missing ECG/segments/0")
        else:
            for grp in ["signal", "beat_annotation", "fiducial_point", "fiducial_feature"]:
                if grp not in seg0:
                    issues.append(f"segment/0 missing: {grp}")
            if "signal" in seg0 and seg0["signal"].shape[0] != 12:
                issues.append(f"signal shape[0]={seg0['signal'].shape[0]} != 12")

    if issues:
        print(f"\n❌ 문제 {len(issues)}개:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\n✅ 구조 정상")


def test_one(dataset_name, cfg, args):
    """CSV에서 1행만 골라 변환하고 결과 H5를 출력"""
    df = pd.read_csv(cfg["old_csv"], low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    idx = args.test_idx
    if idx >= len(df):
        print(f"test_idx={idx} 가 CSV 범위({len(df)})를 벗어남")
        return

    row = df.iloc[idx].to_dict()
    print(f"\n테스트 대상: dataset={dataset_name}, row_idx={idx}")
    print(f"  old filepath: {row.get('filepath','')}")

    old_h5_dir  = cfg.get("old_h5_dir", "")
    output_root = cfg["output_root"]
    new_h5_dir  = os.path.join(output_root, "data")
    label_cols  = cfg.get("label_cols", [])

    # Ray 없이 직접 호출 (테스트용)
    result = ray.get(
        convert_one.remote(row, old_h5_dir, new_h5_dir, label_cols, dataset_name, dry_run=False)
    )

    if result is None:
        print("\n❌ 변환 실패 (old H5를 읽을 수 없거나 저장 오류)")
        return

    new_h5_path = os.path.join(new_h5_dir, os.path.basename(result["filepath"]))
    print(f"  new filepath: {new_h5_path}")
    print(f"  new oid:      {result['oid']}")
    print(f"  new file_name:{result['filepath']}")

    verify_h5(new_h5_path)

    print("\n[CSV row 미리보기]")
    for k, v in result.items():
        if k not in ("nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis", "bs_corr", "bs_dtw"):
            print(f"  {k}: {v}")


# ─────────────────────────────────────────────────────────────
# 데이터셋 단위 처리
# ─────────────────────────────────────────────────────────────
def process_dataset(dataset_name, cfg, args):
    logging.info(f"\n{'='*60}")
    logging.info(f"  {dataset_name} 변환 시작")
    logging.info(f"{'='*60}")

    df = pd.read_csv(cfg["old_csv"], low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    logging.info(f"  CSV 행 수: {len(df):,}")

    old_h5_dir   = cfg.get("old_h5_dir", "")
    output_root  = cfg["output_root"]
    new_h5_dir   = os.path.join(output_root, "data")
    label_cols   = cfg.get("label_cols", [])
    os.makedirs(new_h5_dir, exist_ok=True)

    # 증분 변환: 이미 변환된 파일 스킵
    existing = {
        os.path.splitext(f)[0]
        for f in os.listdir(new_h5_dir)
        if f.endswith(".h5")
    }
    logging.info(f"  기존 변환 파일: {len(existing):,}개")

    rows  = df.to_dict("records")
    total = len(rows)

    def new_stem(row):
        """이 row의 새 file_name stem (prefix 적용)"""
        pfx = get_prefix(dataset_name, str(row.get("filepath", "")))
        return f"{pfx}{row.get('pid','')}{row.get('rid','')}"

    pbar       = tqdm(total=total, desc=f"  {dataset_name}", unit="rec")
    all_rows   = []
    skip_count = 0
    fail_count = 0

    batch_size = args.batch_size
    for b in range((total + batch_size - 1) // batch_size):
        batch   = rows[b * batch_size: (b + 1) * batch_size]
        futures = []
        n_skip  = 0

        for row in batch:
            if new_stem(row) in existing:
                n_skip += 1
                continue
            futures.append(
                convert_one.remote(
                    row, old_h5_dir, new_h5_dir, label_cols, dataset_name, args.dry_run
                )
            )

        pbar.update(n_skip)
        skip_count += n_skip

        for fut in futures:
            result = ray.get(fut)
            pbar.update(1)
            if result is not None:
                all_rows.append(result)
            else:
                fail_count += 1

    pbar.close()

    # 이미 변환된 행도 CSV에 포함 (증분 실행 시 기존 CSV 합산)
    out_csv = os.path.join(output_root, f"{dataset_name}_table.csv")
    if os.path.exists(out_csv) and skip_count > 0:
        df_old = pd.read_csv(out_csv, low_memory=False)
        df_old = df_old.loc[:, ~df_old.columns.str.startswith("Unnamed")]
        combined = pd.concat([df_old, pd.DataFrame(all_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(all_rows)

    combined.to_csv(out_csv, index=False)
    logging.info(
        f"  완료: {len(all_rows):,}개 변환, {skip_count:,}개 기존, {fail_count:,}개 실패 → {out_csv}"
    )
    return combined


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="기존 H5 → 새 표준 H5 변환")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
    )
    parser.add_argument("--num_cpus",   type=int,  default=32)
    parser.add_argument("--batch_size", type=int,  default=2000)
    parser.add_argument("--dry_run",    action="store_true", help="H5 저장 생략, CSV만 생성")
    parser.add_argument("--test",       action="store_true", help="CSV 첫 행 1건만 변환 후 구조 출력")
    parser.add_argument("--test_idx",   type=int, default=0, help="테스트할 CSV 행 인덱스 (기본: 0)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("convert_old_to_new.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"변환 시작: dataset={args.dataset}, cpus={args.num_cpus}, dry_run={args.dry_run}")

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True, _temp_dir="/home/irteam/local-node-d/.temp/ray")

    if args.test:
        ds = args.dataset if args.dataset != "all" else "code15"
        test_one(ds, DATASET_CONFIGS[ds], args)
    else:
        targets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
        for ds in targets:
            process_dataset(ds, DATASET_CONFIGS[ds], args)

    ray.shutdown()
    logging.info("전체 변환 완료")


if __name__ == "__main__":
    main()