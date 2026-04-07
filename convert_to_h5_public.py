"""
Public ECG Dataset → H5 변환 파이프라인
========================================
그룹 구성:
  physionet (8개):
    chapman_shaoxing  (prefix: psh)
    cpsc_2018         (prefix: pcp)
    cpsc_2018_extra   (prefix: pce)
    georgia           (prefix: pge)
    ningbo            (prefix: pnb)
    ptb               (prefix: ppt)
    ptb-xl            (prefix: ppx)
    st_petersburg_incart (prefix: pin)

  zzu (1개):
    ZZU-pECG          (prefix: zzu)

출력 구조:
  output_root/
  ├── public_ecg_table.csv
  ├── {dataset}_table.csv
  ├── file_name.csv
  ├── conversion_public.log
  └── data/                          ← 모든 데이터셋 flat 저장
      ├── psh{pid}{rid}.h5           ← chapman
      ├── pcp{pid}{rid}.h5           ← cpsc2018
      ├── ...
      └── zzu{pid}{rid}.h5           ← zzu_pecg

파일명 규칙:
  file_name : {prefix}{pid}{rid}
  oid       : {prefix}{pid}{rid}{sid}
  filepath  : data/{prefix}{pid}{rid}.h5

실행:
  # physionet 8개
  python convert_to_h5_public.py --group physionet \\
      --physionet_root /home/irteam/ddn-opendata1/raw/physionet.org/files \\
      --output_root    /home/irteam/opendata1/h5/physionet/v2.0

  # ZZU
  python convert_to_h5_public.py --group zzu \\
      --zzu_root    /home/irteam/ddn-opendata1/raw/ZZU-pECG \\
      --output_root /home/irteam/opendata1/h5/zzu/v2.0

  # 특정 데이터셋만
  python convert_to_h5_public.py --dataset georgia,ptbxl \\
      --physionet_root ... --output_root ...

신호 품질은 기본 비활성화. 변환 후 별도 실행:
  python append_signal_quality.py \\
      --csv     OUTPUT_ROOT/public_ecg_table.csv \\
      --h5_root OUTPUT_ROOT

canonical 채널 인덱스 (clinical_ts 기준):
  {i:0, ii:1, iii:2, avr:3, avl:4, avf:5, v1:6, v2:7, v3:8, v4:9, v5:10, v6:11}
TARGET_SIG_NAME → canonical 인덱스:
  [I,  II, III, V1, V2, V3, V4, V5, V6, aVF, aVL, aVR]
  [0,  1,  2,   6,  7,  8,  9, 10, 11,  5,   4,   3 ]
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

HEEDB_DIR = str(Path(__file__).resolve().parent / "heedb")
sys.path.insert(0, HEEDB_DIR)

from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME

# ═══════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════
CANONICAL_TO_TARGET_IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11, 5, 4, 3]
TARGET_SET     = set(TARGET_SIG_NAME)
CHALLENGE_BASE = "challenge-2021/1.0.3/training"

PHYSIONET_DATASETS = [
    "chapman", "cpsc2018", "cpsc_extra", "georgia",
    "ningbo", "ptb", "ptbxl", "stpetersburg",
]
ZZU_DATASETS = ["zzu_pecg"]

LEAD_ALIASES = {
    "DI": "I",    "DII": "II",   "DIII": "III",
    "AVR": "aVR", "AVL": "aVL",  "AVF": "aVF",
    "avr": "aVR", "avl": "aVL",  "avf": "aVF",
    "i": "I",     "ii": "II",    "iii": "III",
    "v1": "V1",   "v2": "V2",    "v3": "V3",
    "v4": "V4",   "v5": "V5",    "v6": "V6",
}

# CSV 컬럼 순서 (heedb_table.csv 스펙)
TABLE_COLS = [
    "filepath", "pid", "rid", "sid", "oid",
    "age", "gender", "height", "weight",
    "fs", "channel_name",
    "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
    "bs_corr", "bs_dtw",
]
FILENAME_COLS = [
    "dataset", "original_filename", "original_filepath",
    "h5_filename", "h5_filepath",
]


# ═══════════════════════════════════════════════════════════════
# age / gender 헬퍼
# ═══════════════════════════════════════════════════════════════
def _default_age(row: dict) -> float:
    try:
        v = float(row.get("age", -1))
        if 0.0 <= v <= 1.5:
            return round(v, 6)
        if 0 < v < 150:
            return round(v / 100.0, 6)
        return -1.0
    except (TypeError, ValueError):
        return -1.0


def _default_gender(row: dict) -> int:
    try:
        v = row.get("sex", row.get("gender", None))
        if v is None:
            return 0
        v = str(v).strip().lower()
        if v in ("1", "male", "m"):
            return 1
        if v in ("0", "-1", "female", "f"):
            return -1
        return 0
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════
# 레코드 목록 생성
# ═══════════════════════════════════════════════════════════════
def _scan_challenge(ds_dir: str) -> list:
    paths = sorted(p[:-4] for p in glob.glob(os.path.join(ds_dir, "g*", "*.hea")))
    return [
        {"record_path": p, "pid": os.path.basename(p), "rid": rid, "age": -1.0, "gender": 0}
        for rid, p in enumerate(paths)
    ]


def _get_zzu_records(zzu_root: str) -> list:
    csv_path = os.path.join(zzu_root, "AttributesDictionary.csv")
    if not os.path.exists(csv_path):
        logging.warning(f"  ECGCode.csv 없음: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    records = []
    # iterrows() 사용: ICD-10 code 등 특수문자 컬럼이 있어 itertuples() 컬럼명 변환 방지
    for rid, row in df.iterrows():
        rel      = str(row.get("Filename", "")).strip()
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
        records.append({"record_path": rec_path, "pid": pid, "rid": rid, "age": age, "gender": gender})
    return records


def _build_configs(physionet_root: str, zzu_root: str) -> dict:
    pb = os.path.join(physionet_root, CHALLENGE_BASE)
    return {
        # ── physionet ──────────────────────────────────────────
        "chapman":      {"name": "Chapman-Shaoxing",       "prefix": "psh", "group": "physionet",
                         "records_fn": lambda: _scan_challenge(os.path.join(pb, "chapman_shaoxing"))},
        "cpsc2018":     {"name": "CPSC 2018",              "prefix": "pcp", "group": "physionet",
                         "records_fn": lambda: _scan_challenge(os.path.join(pb, "cpsc_2018"))},
        "cpsc_extra":   {"name": "CPSC-Extra",             "prefix": "pce", "group": "physionet",
                         "records_fn": lambda: _scan_challenge(os.path.join(pb, "cpsc_2018_extra"))},
        "georgia":      {"name": "Georgia",                "prefix": "pge", "group": "physionet",
                         "records_fn": lambda: _scan_challenge(os.path.join(pb, "georgia"))},
        "ningbo":       {"name": "Ningbo",                 "prefix": "pnb", "group": "physionet",
                         "records_fn": lambda: _scan_challenge(os.path.join(pb, "ningbo"))},
        "ptb":          {"name": "PTB",                    "prefix": "ppt", "group": "physionet",
                         "records_fn": lambda: _scan_challenge(os.path.join(pb, "ptb"))},
        "ptbxl":        {"name": "PTB-XL",                "prefix": "ppx", "group": "physionet",
                         "records_fn": lambda: _scan_challenge(os.path.join(pb, "ptb-xl"))},
        "stpetersburg": {"name": "St. Petersburg INCART",  "prefix": "pin", "group": "physionet",
                         "records_fn": lambda: _scan_challenge(os.path.join(pb, "st_petersburg_incart"))},
        # ── zzu ───────────────────────────────────────────────
        "zzu_pecg":     {"name": "ZZU pECG",              "prefix": "zzu", "group": "zzu",
                         "records_fn": lambda: _get_zzu_records(zzu_root)},
    }


# ═══════════════════════════════════════════════════════════════
# Ray remote: 레코드 1개 → H5 저장
# ═══════════════════════════════════════════════════════════════
@ray.remote
def process_one(
    rec_info:         dict,
    dataset_name:     str,
    prefix:           str,
    h5_dir:           str,
    heedb_dir:        str,
    compute_beat:     bool,
    compute_fiducial: bool,
):
    import os, sys, h5py
    import numpy as np
    import wfdb
    sys.path.insert(0, heedb_dir)
    from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME
    from utils_heedb import extract_beat_annotation, extract_fiducial

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

    sig = rec.p_signal
    if sig is None:
        sig = rec.d_signal
    if sig is None:
        return None

    d         = rec.__dict__
    fs        = d["fs"]
    sig_len   = d["sig_len"]
    raw_names = [_ALIASES.get(n, n) for n in rec.sig_name]

    # 스킵 조건
    if d["n_sig"] != 12:                              return None
    if set(raw_names) != _TARGET_SET:                 return None
    if sig.shape[0] / fs < 1.0:                      return None
    if np.any(np.all(sig == 0, axis=0)):              return None

    # WFDB 헤더에서 age / gender 보완
    age    = rec_info.get("age",    -1.0)
    gender = rec_info.get("gender",  0)
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
    idx           = [raw_names.index(n) for n in TARGET_SIG_NAME]
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
            ba          = extract_beat_annotation(np.nan_to_num(sig_reordered[1].astype(np.float32)), fs)
            ba_list     = [ba]
            beat_method = "neurokit2"
        except Exception:
            pass

    # fiducial (옵션)
    fp_list, ff_list, fidu_method = None, None, ""
    if compute_fiducial:
        try:
            fp, ff      = extract_fiducial(np.nan_to_num(sig_reordered.astype(np.float32)), fs)
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
                file_name           = file_name,
                beat_ext_method     = beat_method,
                fidu_extract_method = fidu_method,
                record_name         = d["record_name"],
                n_sig=12, fs=fs, sig_len=sig_len,
                base_time           = str(d.get("base_time", "") or ""),
                base_date           = str(d.get("base_date", "") or ""),
                sig_name            = TARGET_SIG_NAME,
                fmt=fmt_r, adc_gain=gain_r, baseline=bl_r,
                units=units_r, adc_res=res_r, adc_zero=zero_r,
                signal              = [sig_reordered],
                seg_len             = 1,
                beat_annotation     = ba_list,
                fiducial_point      = fp_list,
                fiducial_feature    = ff_list,
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
    # 모든 데이터셋을 data/ 아래 flat하게 저장 (prefix로 구분)
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

    # prefix로 필터링 → 해당 데이터셋 파일만 기존으로 인식
    existing = {f[:-3] for f in os.listdir(h5_dir)
                if f.endswith(".h5") and f.startswith(prefix)}
    todo     = [r for r in records if f"{prefix}{r['pid']}{r['rid']}" not in existing]
    logging.info(f"  기존: {len(existing):,}개 | 대기: {len(todo):,}개")

    all_rows = []
    with tqdm(total=len(todo), desc=f"  {cfg['name']}", unit="rec") as pbar:
        for i in range(0, len(todo), args.batch_size):
            batch   = todo[i:i + args.batch_size]
            futures = [
                process_one.remote(
                    r, dataset_name, prefix, str(h5_dir), HEEDB_DIR,
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
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Public ECG → HEEDB H5 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
그룹 실행 예시:
  # physionet 8개 → physionet/v2.0
  python convert_to_h5_public.py --group physionet \\
      --physionet_root /home/irteam/ddn-opendata1/raw/physionet.org/files \\
      --output_root    /home/irteam/opendata1/h5/physionet/v2.0 \\
      --num_cpus 64

  # ZZU → zzu/v2.0
  python convert_to_h5_public.py --group zzu \\
      --zzu_root    /home/irteam/ddn-opendata1/raw/ZZU-pECG \\
      --output_root /home/irteam/opendata1/h5/zzu/v2.0 \\
      --num_cpus 64

  # 특정 데이터셋만
  python convert_to_h5_public.py --dataset georgia,ptbxl \\
      --physionet_root ... --output_root ...
        """,
    )
    # 대상 선택 (--group 또는 --dataset 중 하나)
    target_grp = parser.add_mutually_exclusive_group()
    target_grp.add_argument("--group",   type=str, choices=["physionet", "zzu", "all"],
                            help="physionet | zzu | all")
    target_grp.add_argument("--dataset", type=str,
                            help="쉼표 구분: chapman,georgia,zzu_pecg")

    parser.add_argument("--physionet_root", type=str,
                        default="/home/irteam/ddn-opendata1/raw/physionet.org/files")
    parser.add_argument("--zzu_root",       type=str,
                        default="/home/irteam/ddn-opendata1/raw/ZZU-pECG")
    parser.add_argument("--output_root",    type=str, required=True)
    parser.add_argument("--num_cpus",       type=int, default=64)
    parser.add_argument("--batch_size",     type=int, default=2000)
    parser.add_argument("--compute_beat",     action="store_true")
    parser.add_argument("--compute_fiducial", action="store_true")
    parser.add_argument("--compute_quality",  action="store_true",
                        help="신호 품질 계산 (기본 OFF; append_signal_quality.py 권장)")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    log_path = output_root / "conversion_public.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    configs = _build_configs(args.physionet_root, args.zzu_root)

    # 대상 데이터셋 결정
    if args.group == "physionet":
        target_datasets = PHYSIONET_DATASETS
    elif args.group == "zzu":
        target_datasets = ZZU_DATASETS
    elif args.group == "all":
        target_datasets = PHYSIONET_DATASETS + ZZU_DATASETS
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

    if args.compute_quality:
        logging.warning("--compute_quality 는 변환 속도를 크게 낮춥니다. append_signal_quality.py 권장.")

    logging.info(f"그룹/대상    : {args.group or args.dataset} → {target_datasets}")
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
    table_path = output_root / "public_ecg_table.csv"
    _make_table_df(all_rows).to_csv(table_path, index=False)
    logging.info(f"\npublic_ecg_table.csv : {len(all_rows):,}행 → {table_path}")

    # file_name.csv
    fn_path = output_root / "file_name.csv"
    _make_filename_df(all_rows).to_csv(fn_path, index=False)
    logging.info(f"file_name.csv        : {len(all_rows):,}행 → {fn_path}")

    logging.info(f"\n신호 품질 계산:")
    logging.info(f"  python append_signal_quality.py \\")
    logging.info(f"      --csv {table_path} --h5_root {output_root}")
    logging.info("전체 완료")


if __name__ == "__main__":
    main()