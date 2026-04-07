"""
통합 ECG → H5 변환 파이프라인
================================
HEEDB (I0001/I0006) + 공개 데이터셋 (physionet 8종 + ZZU) 을 동일한 H5 구조로 변환합니다.

그룹:
  heedb      : I0001 (MGH, prefix=he1), I0006 (EUH, prefix=he6)
  physionet  : chapman(psh), cpsc2018(pcp), cpsc_extra(pce), georgia(pge),
               ningbo(pnb), ptb(ppt), ptbxl(ppx), stpetersburg(pin)
  zzu        : ZZU-pECG(zzu)

실행 예시:
  # HEEDB 전체
  python convert_to_h5.py --group heedb --output_root /data/h5/heedb/v4.0

  # physionet 전체
  python convert_to_h5.py --group physionet \
      --physionet_root /data/raw/physionet.org/files \
      --output_root /data/h5/physionet/v2.0

  # ZZU
  python convert_to_h5.py --group zzu \
      --zzu_root /data/raw/ZZU-pECG \
      --output_root /data/h5/zzu/v2.0

  # 전체
  python convert_to_h5.py --group all \
      --heedb_root /data/raw/heedb/ECG \
      --physionet_root /data/raw/physionet.org/files \
      --zzu_root /data/raw/ZZU-pECG \
      --output_root /data/h5/all/v1.0

  # 특정 데이터셋만
  python convert_to_h5.py --dataset georgia,ptbxl,heedb_i0001 \
      --physionet_root ... --heedb_root ... --output_root ...
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np
import pandas as pd
import h5py
import ray
from pathlib import Path
from tqdm import tqdm

# utils/ 를 import 할 수 있도록 스크립트 위치 기준 경로 추가
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, SCRIPT_DIR)

from utils.h5_structure import create_h5_structure, TARGET_SIG_NAME

# ═══════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════
TARGET_SET     = set(TARGET_SIG_NAME)
CHALLENGE_BASE = "challenge-2021/1.0.3/training"

LEAD_ALIASES = {
    "DI": "I",    "DII": "II",   "DIII": "III",
    "AVR": "aVR", "AVL": "aVL",  "AVF": "aVF",
    "avr": "aVR", "avl": "aVL",  "avf": "aVF",
    "i": "I",     "ii": "II",    "iii": "III",
    "v1": "V1",   "v2": "V2",    "v3": "V3",
    "v4": "V4",   "v5": "V5",    "v6": "V6",
}

PHYSIONET_DATASETS = [
    "chapman", "cpsc2018", "cpsc_extra", "georgia",
    "ningbo", "ptb", "ptbxl", "stpetersburg",
]
ZZU_DATASETS    = ["zzu_pecg"]
HEEDB_DATASETS  = ["heedb_i0001", "heedb_i0006"]

TABLE_COLS = [
    "filepath", "dataset", "pid", "rid", "sid", "oid",
    "age", "gender", "height", "weight",
    "fs", "channel_name",
    "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
    "bs_corr", "bs_dtw",
]
FILENAME_COLS = [
    "dataset", "original_filename", "original_filepath",
    "h5_filename", "h5_filepath",
]

# HEEDB combined metadata 컬럼
COMBINED_COLUMNS = [
    "BDSPPatientID", "FileName", "FileID", "PatientRace", "Sex",
    "ECGAcquisitionTime", "DateOfBirth", "LastKnownVisitDate",
    "DateOfDeath", "AgeAtAcquisition", "AgeAtLastVisit", "AgeAtDeath",
    "source",
]


# ═══════════════════════════════════════════════════════════════
# 공통 헬퍼
# ═══════════════════════════════════════════════════════════════
def encode_gender(val):
    if not isinstance(val, str):
        return 0
    v = val.strip().upper()
    if v in ("MALE", "M", "1"):
        return 1
    if v in ("FEMALE", "F", "0", "-1"):
        return -1
    return 0


# ═══════════════════════════════════════════════════════════════
# 레코드 목록 생성 — 공개 데이터셋
# ═══════════════════════════════════════════════════════════════
def _scan_challenge(ds_dir: str) -> list:
    paths = sorted(p[:-4] for p in glob.glob(os.path.join(ds_dir, "g*", "*.hea")))
    return [
        {"record_path": p, "pid": os.path.basename(p), "rid": rid,
         "age": -1.0, "gender": 0, "source": "wfdb"}
        for rid, p in enumerate(paths)
    ]


def _get_zzu_records(zzu_root: str) -> list:
    csv_path = os.path.join(zzu_root, "AttributesDictionary.csv")
    if not os.path.exists(csv_path):
        logging.warning(f"  AttributesDictionary.csv 없음: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    records = []
    for rid, row in df.iterrows():
        rel = str(row.get("Filename", "")).strip()
        if not rel:
            continue
        rec_path = os.path.join(zzu_root, "Child_ecg", rel)
        try:
            age_days = float(str(row.get("Age", "")).rstrip("d"))
            age = round((age_days / 365.25) / 100.0, 6)
        except (ValueError, AttributeError):
            age = -1.0
        gender_raw = str(row.get("Gender", "")).strip().strip("'").lower()
        gender = 1 if gender_raw == "male" else (-1 if gender_raw == "female" else 0)
        pid = str(row.get("Patient_ID", os.path.basename(rel))).strip()
        records.append({"record_path": rec_path, "pid": pid, "rid": rid,
                        "age": age, "gender": gender, "source": "wfdb"})
    return records


# ═══════════════════════════════════════════════════════════════
# 레코드 목록 생성 — HEEDB
# ═══════════════════════════════════════════════════════════════
def _get_heedb_records(base_dir: str, gender_field: str) -> list:
    wfdb_root = os.path.join(base_dir, "WFDB")
    meta_path = os.path.join(base_dir, "metadata", "metadata.csv")
    if not os.path.exists(meta_path):
        logging.warning(f"  metadata.csv 없음: {meta_path}")
        return []
    df = pd.read_csv(meta_path, dtype=str, low_memory=False)
    records = []
    for rid, row in df.iterrows():
        pid = str(row.get("BDSPPatientID", "")).strip()
        fn_raw = str(row.get("FileName", "")).strip().lstrip("/")
        fn_clean = fn_raw[5:] if fn_raw.startswith("WFDB/") else fn_raw
        if not pid or pid == "nan":
            continue
        try:
            age = round(float(row.get("AgeAtAcquisition", -1)) / 365.25 / 100.0, 6)
        except (TypeError, ValueError):
            age = -1.0
        gender = encode_gender(str(row.get(gender_field, "")))
        records.append({
            "record_path": os.path.join(wfdb_root, fn_clean),
            "pid": pid, "rid": rid,
            "age": age, "gender": gender,
            "source": "heedb",
            "_row_dict": dict(row),
            "_gender_field": gender_field,
        })
    return records


def _build_all_configs(args) -> dict:
    """모든 데이터셋의 config dict 를 생성합니다."""
    configs = {}

    # physionet
    if args.physionet_root:
        pb = os.path.join(args.physionet_root, CHALLENGE_BASE)
        for ds, subdir in [
            ("chapman",      "chapman_shaoxing"),
            ("cpsc2018",     "cpsc_2018"),
            ("cpsc_extra",   "cpsc_2018_extra"),
            ("georgia",      "georgia"),
            ("ningbo",       "ningbo"),
            ("ptb",          "ptb"),
            ("ptbxl",        "ptb-xl"),
            ("stpetersburg", "st_petersburg_incart"),
        ]:
            d = os.path.join(pb, subdir)
            configs[ds] = {
                "name": ds, "prefix": {"chapman":"psh","cpsc2018":"pcp","cpsc_extra":"pce",
                    "georgia":"pge","ningbo":"pnb","ptb":"ppt","ptbxl":"ppx","stpetersburg":"pin"}[ds],
                "group": "physionet",
                "records_fn": lambda _d=d: _scan_challenge(_d),
            }

    # zzu
    if args.zzu_root:
        configs["zzu_pecg"] = {
            "name": "zzu_pecg", "prefix": "zzu", "group": "zzu",
            "records_fn": lambda: _get_zzu_records(args.zzu_root),
        }

    # heedb
    if args.heedb_root:
        for inst_code, inst_name, prefix, gf in [
            ("I0001", "heedb_i0001", "he1", "SexDSC"),
            ("I0006", "heedb_i0006", "he6", "Sex"),
        ]:
            bd = os.path.join(args.heedb_root, inst_code)
            configs[inst_name] = {
                "name": inst_name, "prefix": prefix, "group": "heedb",
                "records_fn": lambda _bd=bd, _gf=gf: _get_heedb_records(_bd, _gf),
                "inst_code": inst_code,
            }

    return configs


# ═══════════════════════════════════════════════════════════════
# Ray remote: 레코드 1개 → H5 저장
# ═══════════════════════════════════════════════════════════════
@ray.remote
def process_one(
    rec_info:         dict,
    dataset_name:     str,
    prefix:           str,
    h5_dir:           str,
    script_dir:       str,
    compute_beat:     bool,
    compute_fiducial: bool,
):
    import os, sys, h5py
    import numpy as np
    import wfdb
    sys.path.insert(0, script_dir)
    from utils.h5_structure import create_h5_structure, TARGET_SIG_NAME
    from utils.signal_processing import (
        reorder_signal, has_zero_lead,
        extract_beat_annotation, extract_fiducial,
    )

    _ALIASES = {
        "DI": "I",   "DII": "II",  "DIII": "III",
        "AVR": "aVR","AVL": "aVL", "AVF": "aVF",
        "avr": "aVR","avl": "aVL", "avf": "aVF",
        "i": "I",    "ii": "II",   "iii": "III",
        "v1": "V1",  "v2": "V2",   "v3": "V3",
        "v4": "V4",  "v5": "V5",   "v6": "V6",
    }
    _TARGET_SET = set(TARGET_SIG_NAME)

    # WFDB 로드
    try:
        rec = wfdb.rdrecord(rec_info["record_path"])
    except Exception:
        return None

    sig = rec.p_signal if rec.p_signal is not None else rec.d_signal
    if sig is None:
        return None

    d         = rec.__dict__
    fs        = d["fs"]
    sig_len   = d["sig_len"]
    raw_names = [_ALIASES.get(n, n) for n in rec.sig_name]

    # 스킵 조건
    if d["n_sig"] != 12:              return None
    if set(raw_names) != _TARGET_SET: return None
    if sig.shape[0] / fs < 1.0:      return None
    if has_zero_lead(sig):            return None

    # age / gender — HEEDB는 rec_info에 이미 설정됨, 공개 데이터셋은 헤더 보완
    age    = rec_info.get("age",    -1.0)
    gender = rec_info.get("gender",  0)
    if rec_info.get("source") == "wfdb":
        # 공개 데이터셋: WFDB 헤더에서 age/gender 보완
        if age == -1.0 or gender == 0:
            for c in (getattr(rec, "comments", []) or []):
                cl = c.strip().lstrip("#").strip().lower()
                if age == -1.0 and cl.startswith("age:"):
                    try:
                        v = float(cl.split(":", 1)[1].strip().split()[0])
                        if 0 < v < 150:
                            age = round(v / 100.0, 6)
                    except Exception:
                        pass
                if gender == 0 and (cl.startswith("sex:") or cl.startswith("gender:")):
                    v = cl.split(":", 1)[1].strip().split()[0].lower()
                    gender = 1 if v in ("male", "m", "1") else (-1 if v in ("female", "f", "0") else 0)

    # reorder → (12, samples) fp16
    idx = [raw_names.index(n) for n in TARGET_SIG_NAME]
    sig_reordered = sig[:, idx].T.astype(np.float16)

    fmt_r   = [d["fmt"][i]      for i in idx] if d.get("fmt")      else None
    gain_r  = [d["adc_gain"][i] for i in idx] if d.get("adc_gain") else None
    bl_r    = [d["baseline"][i] for i in idx] if d.get("baseline") else None
    units_r = [d["units"][i]    for i in idx] if d.get("units")    else None
    res_r   = [d["adc_res"][i]  for i in idx] if d.get("adc_res")  else None
    zero_r  = [d["adc_zero"][i] for i in idx] if d.get("adc_zero") else None

    pid       = rec_info["pid"]
    rid       = rec_info["rid"]
    sid       = 0
    file_name = f"{prefix}{pid}{rid}"
    oid       = f"{prefix}{pid}{rid}{sid}"
    h5_path   = os.path.join(h5_dir, f"{file_name}.h5")

    # beat_annotation (옵션)
    ba_list, beat_method = None, ""
    if compute_beat:
        try:
            ba = extract_beat_annotation(
                np.nan_to_num(sig_reordered[1].astype(np.float32)), fs
            )
            ba_list     = [ba]
            beat_method = "neurokit2"
        except Exception:
            pass

    # fiducial (옵션)
    fp_list, ff_list, fidu_method = None, None, ""
    if compute_fiducial:
        try:
            fp, ff = extract_fiducial(
                np.nan_to_num(sig_reordered.astype(np.float32)), fs
            )
            fp_list     = [fp]
            ff_list     = [ff]
            fidu_method = "neurokit2-dwt"
        except Exception:
            pass

    # H5 저장
    try:
        with h5py.File(h5_path, "w") as h5f:
            create_h5_structure(
                h5f,
                file_name=file_name,
                beat_ext_method=beat_method,
                fidu_extract_method=fidu_method,
                record_name=d["record_name"],
                n_sig=12, fs=fs, sig_len=sig_len,
                base_time=str(d.get("base_time", "") or ""),
                base_date=str(d.get("base_date", "") or ""),
                sig_name=TARGET_SIG_NAME,
                fmt=fmt_r, adc_gain=gain_r, baseline=bl_r,
                units=units_r, adc_res=res_r, adc_zero=zero_r,
                signal=[sig_reordered], seg_len=1,
                beat_annotation=ba_list,
                fiducial_point=fp_list,
                fiducial_feature=ff_list,
            )
    except Exception:
        if os.path.exists(h5_path):
            os.remove(h5_path)
        return None

    return {
        "filepath":      f"data/{file_name}.h5",
        "dataset":       dataset_name,
        "pid":           pid,
        "rid":           rid,
        "sid":           sid,
        "oid":           oid,
        "age":           age,
        "gender":        gender,
        "height":        np.nan,
        "weight":        np.nan,
        "fs":            fs,
        "channel_name":  str(TARGET_SIG_NAME),
        "_record_name":  d["record_name"],
        "_record_path":  rec_info["record_path"],
        "_h5_filename":  f"{file_name}.h5",
    }


# ═══════════════════════════════════════════════════════════════
# CSV 헬퍼
# ═══════════════════════════════════════════════════════════════
def _make_table_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ["nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis", "bs_corr", "bs_dtw"]:
        if col not in df.columns:
            df[col] = ""
    drop = [c for c in df.columns if c.startswith("_")]
    df   = df.drop(columns=drop)
    keep = [c for c in TABLE_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in TABLE_COLS]
    return df[keep + extra]


def _make_filename_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(
        [{"dataset":           r["dataset"],
          "original_filename": r["_record_name"],
          "original_filepath": r["_record_path"],
          "h5_filename":       r["_h5_filename"],
          "h5_filepath":       r["filepath"]}
         for r in rows],
        columns=FILENAME_COLS,
    )


# ═══════════════════════════════════════════════════════════════
# 데이터셋 1개 처리
# ═══════════════════════════════════════════════════════════════
def process_dataset(dataset_name: str, cfg: dict, output_root: Path, args) -> list:
    prefix = cfg["prefix"]
    h5_dir = output_root / "data"
    os.makedirs(h5_dir, exist_ok=True)

    logging.info(f"\n{'='*60}")
    logging.info(f"  {cfg['name']} (prefix={prefix})")
    logging.info(f"{'='*60}")

    try:
        records = cfg["records_fn"]()
    except Exception as e:
        logging.error(f"  레코드 목록 생성 실패: {e}")
        return []

    if not records:
        logging.warning(f"  [{cfg['name']}] 레코드 없음, 스킵")
        return []

    logging.info(f"  전체 레코드: {len(records):,}개")

    # 기존 파일 스킵 (prefix로 필터)
    existing = {f[:-3] for f in os.listdir(h5_dir)
                if f.endswith(".h5") and f.startswith(prefix)}
    todo = [r for r in records if f"{prefix}{r['pid']}{r['rid']}" not in existing]
    logging.info(f"  기존: {len(existing):,}개 | 대기: {len(todo):,}개")

    # Ray remote에 전달하기 전에 직렬화 불가한 필드 제거
    for r in todo:
        r.pop("_row_dict", None)
        r.pop("_gender_field", None)

    all_rows = []
    with tqdm(total=len(todo), desc=f"  {cfg['name']}", unit="rec") as pbar:
        for i in range(0, len(todo), args.batch_size):
            batch   = todo[i:i + args.batch_size]
            futures = [
                process_one.remote(
                    r, dataset_name, prefix, str(h5_dir), SCRIPT_DIR,
                    args.compute_beat, args.compute_fiducial,
                )
                for r in batch
            ]
            for res in ray.get(futures):
                pbar.update(1)
                if res is not None:
                    all_rows.append(res)

    logging.info(f"  완료: {len(all_rows):,}개 저장 | {len(todo)-len(all_rows):,}개 스킵/실패")
    return all_rows


# ═══════════════════════════════════════════════════════════════
# HEEDB combined metadata 저장
# ═══════════════════════════════════════════════════════════════
def _save_heedb_combined_metadata(args, output_root: Path, target_datasets: list):
    """HEEDB 기관별 metadata.csv를 통합하여 combined_metadata.csv로 저장"""
    if not args.heedb_root:
        return
    all_dfs = []
    for inst_name in target_datasets:
        if not inst_name.startswith("heedb_"):
            continue
        inst_code = inst_name.replace("heedb_", "").upper()  # i0001 → I0001
        meta_path = os.path.join(args.heedb_root, inst_code, "metadata", "metadata.csv")
        if not os.path.exists(meta_path):
            continue
        df = pd.read_csv(meta_path, dtype=str, low_memory=False)
        if "SexDSC" in df.columns and "Sex" not in df.columns:
            df["Sex"] = df["SexDSC"]
        df["source"] = inst_code
        keep = [c for c in COMBINED_COLUMNS if c in df.columns]
        all_dfs.append(df[keep])
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_root / "combined_metadata.csv"
        combined.to_csv(combined_path, index=False)
        logging.info(f"  combined_metadata.csv: {len(combined):,}행 → {combined_path}")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="통합 ECG → H5 변환 (HEEDB + 공개 데이터셋)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예시:
  # HEEDB 전체
  python convert_to_h5.py --group heedb \\
      --heedb_root /data/raw/heedb/ECG \\
      --output_root /data/h5/heedb/v4.0 --num_cpus 64

  # physionet 전체
  python convert_to_h5.py --group physionet \\
      --physionet_root /data/raw/physionet.org/files \\
      --output_root /data/h5/physionet/v2.0 --num_cpus 64

  # ZZU
  python convert_to_h5.py --group zzu \\
      --zzu_root /data/raw/ZZU-pECG \\
      --output_root /data/h5/zzu/v2.0

  # 전체 (HEEDB + physionet + ZZU)
  python convert_to_h5.py --group all \\
      --heedb_root ... --physionet_root ... --zzu_root ... \\
      --output_root /data/h5/all/v1.0

  # 특정 데이터셋
  python convert_to_h5.py --dataset georgia,ptbxl,heedb_i0001 \\
      --physionet_root ... --heedb_root ... --output_root ...
        """,
    )
    # 대상 선택
    target_grp = parser.add_mutually_exclusive_group()
    target_grp.add_argument("--group", type=str,
                            choices=["heedb", "physionet", "zzu", "all"],
                            help="그룹 단위 변환")
    target_grp.add_argument("--dataset", type=str,
                            help="쉼표 구분 데이터셋명 (예: georgia,ptbxl,heedb_i0001)")

    # 데이터 경로
    parser.add_argument("--heedb_root",    type=str,
                        default="/home/irteam/opendata1/raw/heedb/ECG",
                        help="HEEDB ECG 루트 (I0001, I0006 상위 디렉토리)")
    parser.add_argument("--physionet_root", type=str,
                        default="/home/irteam/ddn-opendata1/raw/physionet.org/files",
                        help="PhysioNet 데이터 루트")
    parser.add_argument("--zzu_root",       type=str,
                        default="/home/irteam/ddn-opendata1/raw/ZZU-pECG",
                        help="ZZU-pECG 데이터 루트")
    parser.add_argument("--output_root",    type=str, required=True,
                        help="H5 출력 루트")

    # 실행 옵션
    parser.add_argument("--num_cpus",         type=int, default=64)
    parser.add_argument("--batch_size",       type=int, default=2000)
    parser.add_argument("--compute_beat",     action="store_true", help="beat_annotation 생성")
    parser.add_argument("--compute_fiducial", action="store_true", help="fiducial_point/feature 생성")

    args = parser.parse_args()
    output_root = Path(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    log_path = output_root / "conversion.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    configs = _build_all_configs(args)

    # 대상 데이터셋 결정
    if args.group == "heedb":
        target_datasets = HEEDB_DATASETS
    elif args.group == "physionet":
        target_datasets = PHYSIONET_DATASETS
    elif args.group == "zzu":
        target_datasets = ZZU_DATASETS
    elif args.group == "all":
        target_datasets = HEEDB_DATASETS + PHYSIONET_DATASETS + ZZU_DATASETS
    elif args.dataset:
        target_datasets = [d.strip() for d in args.dataset.split(",")]
        unknown = [d for d in target_datasets if d not in configs]
        if unknown:
            logging.error(f"알 수 없는 데이터셋: {unknown}")
            logging.error(f"선택 가능: {list(configs.keys())}")
            return
    else:
        parser.print_help()
        return

    # 필요한 root 경로 확인
    missing = []
    for ds in target_datasets:
        if ds not in configs:
            missing.append(ds)
    if missing:
        logging.error(f"config 없음 (경로 미지정?): {missing}")
        return

    logging.info(f"대상        : {target_datasets}")
    logging.info(f"heedb_root  : {args.heedb_root}")
    logging.info(f"physionet   : {args.physionet_root}")
    logging.info(f"zzu_root    : {args.zzu_root}")
    logging.info(f"output_root : {output_root}")
    logging.info(f"옵션        : beat={args.compute_beat}, fiducial={args.compute_fiducial}")

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)
    logging.info(f"Ray CPUs    : {ray.available_resources().get('CPU', 'N/A')}")

    all_rows = []
    for ds_name in target_datasets:
        rows = process_dataset(ds_name, configs[ds_name], output_root, args)
        if not rows:
            continue
        all_rows.extend(rows)

        # 데이터셋별 CSV 즉시 저장
        sub_path = output_root / f"{ds_name}_table.csv"
        _make_table_df(rows).to_csv(sub_path, index=False)
        logging.info(f"  {ds_name}_table.csv: {len(rows):,}행 → {sub_path}")

    ray.shutdown()

    if not all_rows:
        logging.warning("변환된 레코드가 없습니다.")
        return

    # 전체 통합 CSV
    table_path = output_root / "ecg_table.csv"
    _make_table_df(all_rows).to_csv(table_path, index=False)
    logging.info(f"\necg_table.csv    : {len(all_rows):,}행 → {table_path}")

    # file_name.csv
    fn_path = output_root / "file_name.csv"
    _make_filename_df(all_rows).to_csv(fn_path, index=False)
    logging.info(f"file_name.csv    : {len(all_rows):,}행 → {fn_path}")

    # HEEDB combined metadata
    _save_heedb_combined_metadata(args, output_root, target_datasets)

    logging.info(f"\n후속 작업:")
    logging.info(f"  # fiducial 추가")
    logging.info(f"  python append_fiducial.py --csv {table_path} --h5_root {output_root}")
    logging.info(f"  # signal quality 계산")
    logging.info(f"  python append_signal_quality.py --csv {table_path} --h5_root {output_root}")
    logging.info("전체 완료")


if __name__ == "__main__":
    main()