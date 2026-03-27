"""
HEEDB H5 변환 메인 스크립트
============================
I0001 + I0006 → H5 + heedb_table.csv + combined_metadata.csv + file_name.csv

실행:
  python heedb/convert_to_h5_heedb.py
  python heedb/convert_to_h5_heedb.py --num_cpus 64
  python heedb/convert_to_h5_heedb.py --compute_beat --compute_fiducial

신호 품질(signal quality) 계산은 기본적으로 비활성화되어 있습니다.
별도로 계산하려면 프로젝트 루트의 append_signal_quality.py 를 사용하세요:
  python append_signal_quality.py --csv /path/to/heedb_table.csv --h5_root /path/to/h5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import wfdb
import h5py
import ray
import logging
from pathlib import Path
from tqdm import tqdm

# heedb/ 내부 모듈 경로 보장
sys.path.insert(0, str(Path(__file__).resolve().parent))

from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME
from utils_heedb import (
    reorder_signal, has_zero_lead,
    extract_beat_annotation, extract_fiducial,
)

# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════
INSTITUTIONS = [
    {
        "name": "I0001",
        "prefix": "he1",
        "base_dir": "/home/irteam/opendata1/raw/heedb/ECG/I0001",
        "gender_field": "SexDSC",
    },
    {
        "name": "I0006",
        "prefix": "he6",
        "base_dir": "/home/irteam/opendata1/raw/heedb/ECG/I0006",
        "gender_field": "Sex",
    },
]

OUTPUT_ROOT = "/home/irteam/opendata1/h5/heedb/v4.0"
DATA_DIR = os.path.join(OUTPUT_ROOT, "data")

COMBINED_COLUMNS = [
    "BDSPPatientID", "FileName", "FileID", "PatientRace", "Sex",
    "ECGAcquisitionTime", "DateOfBirth", "LastKnownVisitDate",
    "DateOfDeath", "AgeAtAcquisition", "AgeAtLastVisit", "AgeAtDeath",
    "source",
]


def encode_gender(val):
    if not isinstance(val, str):
        return 0
    v = val.strip().upper()
    if v == "MALE":
        return 1
    elif v == "FEMALE":
        return -1
    return 0


# ═══════════════════════════════════════════════════════════════
# Ray remote: 1 레코드 처리 + H5 저장
# ═══════════════════════════════════════════════════════════════
@ray.remote
def process_and_save_one(
    row_dict, rid, wfdb_root, prefix, gender_field,
    h5_dir, compute_beat, compute_fiducial,
):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME
    from utils_heedb import reorder_signal, has_zero_lead, extract_beat_annotation, extract_fiducial

    pid = str(row_dict.get("BDSPPatientID", "")).strip()
    fn_raw = str(row_dict.get("FileName", "")).strip().lstrip("/")
    fn_clean = fn_raw[5:] if fn_raw.startswith("WFDB/") else fn_raw

    if not pid or pid == "nan":
        return None

    try:
        rec = wfdb.rdrecord(os.path.join(wfdb_root, fn_clean))
    except Exception:
        return None

    sig = rec.p_signal if rec.p_signal is not None else rec.d_signal
    if sig is None:
        return None

    d = rec.__dict__
    fs = d["fs"]
    samples = sig.shape[0]
    duration = samples / fs

    if d["n_sig"] != 12:
        return None
    if duration < 1.0:
        return None
    if set(rec.sig_name) != set(TARGET_SIG_NAME):
        return None
    if has_zero_lead(sig):
        return None

    sig_reordered = reorder_signal(sig, rec.sig_name)  # (12, timepoints)

    file_name = f"{prefix}{pid}{rid}"
    sid = 0
    oid = f"{prefix}{pid}{rid}{sid}"

    reorder_idx = [rec.sig_name.index(n) for n in TARGET_SIG_NAME]
    fmt_r  = [d["fmt"][i]      for i in reorder_idx] if d.get("fmt")      else None
    gain_r = [d["adc_gain"][i] for i in reorder_idx] if d.get("adc_gain") else None
    bl_r   = [d["baseline"][i] for i in reorder_idx] if d.get("baseline") else None
    units_r = [d["units"][i]   for i in reorder_idx] if d.get("units")    else None
    res_r  = [d["adc_res"][i]  for i in reorder_idx] if d.get("adc_res")  else None
    zero_r = [d["adc_zero"][i] for i in reorder_idx] if d.get("adc_zero") else None

    ba_list, beat_method = None, ""
    if compute_beat:
        ba = extract_beat_annotation(sig_reordered[1], fs)
        ba_list = [ba]
        beat_method = "neurokit2"

    fp_list, ff_list, fidu_method = None, None, ""
    if compute_fiducial:
        fp, ff = extract_fiducial(sig_reordered, fs)
        fp_list = [fp]
        ff_list = [ff]
        fidu_method = "neurokit2-dwt"

    h5_path = os.path.join(h5_dir, f"{file_name}.h5")
    try:
        with h5py.File(h5_path, "w") as h5f:
            create_h5_structure(
                h5f,
                file_name=file_name,
                beat_ext_method=beat_method,
                fidu_extract_method=fidu_method,
                record_name=d["record_name"],
                n_sig=12, fs=fs, sig_len=d["sig_len"],
                base_time=str(d.get("base_time", "")),
                base_date=str(d.get("base_date", "")),
                sig_name=TARGET_SIG_NAME,
                fmt=fmt_r, adc_gain=gain_r, baseline=bl_r,
                units=units_r, adc_res=res_r, adc_zero=zero_r,
                signal=[sig_reordered], seg_len=1,
                beat_annotation=ba_list,
                fiducial_point=fp_list,
                fiducial_feature=ff_list,
            )
    except Exception:
        return None

    age_raw = row_dict.get("AgeAtAcquisition", None)
    try:
        age = float(age_raw) / 365.25 / 100
    except (TypeError, ValueError):
        age = -1

    gender = encode_gender(row_dict.get(gender_field, ""))

    csv_row = {
        "filepath": f"data/{file_name}.h5",
        "pid": pid, "rid": rid, "sid": sid, "oid": oid,
        "age": round(age, 6) if age != -1 else -1,
        "gender": gender,
        "height": np.nan, "weight": np.nan,
        "fs": fs,
        "channel_name": str(TARGET_SIG_NAME),
    }

    fn_row = {
        "source": "I0001" if prefix == "he1" else "I0006",
        "original_filename": d["record_name"],
        "original_filepath": os.path.join(wfdb_root, fn_clean),
        "h5_filename": f"{file_name}.h5",
        "h5_filepath": f"data/{file_name}.h5",
    }

    return {"csv_row": csv_row, "fn_row": fn_row}


# ═══════════════════════════════════════════════════════════════
# 기관 1개 처리
# ═══════════════════════════════════════════════════════════════
def process_institution(inst, args):
    name       = inst["name"]
    prefix     = inst["prefix"]
    base_dir   = inst["base_dir"]
    gender_field = inst["gender_field"]
    wfdb_root  = os.path.join(base_dir, "WFDB")
    meta_path  = os.path.join(base_dir, "metadata", "metadata.csv")

    logging.info(f"{'='*60}")
    logging.info(f"  {name} 시작")
    logging.info(f"{'='*60}")

    df = pd.read_csv(meta_path, dtype=str, low_memory=False)
    logging.info(f"  metadata: {len(df):,}행")

    existing = set()
    if os.path.exists(DATA_DIR):
        existing = {f[:-3] for f in os.listdir(DATA_DIR) if f.endswith(".h5") and f.startswith(prefix)}
    logging.info(f"  이미 변환: {len(existing):,}개")

    rows = df.to_dict("records")
    total = len(rows)
    pbar = tqdm(total=total, desc=f"  {name} 전체", unit="rec")

    all_csv_rows = []
    all_fn_rows = []
    skip_count = 0
    save_count = 0
    batch_size = args.batch_size
    total_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = rows[start:end]

        futures = []
        batch_existing = 0
        for rid_offset, row in enumerate(batch):
            rid = start + rid_offset
            file_name = f"{prefix}{str(row.get('BDSPPatientID', '')).strip()}{rid}"
            if file_name in existing:
                batch_existing += 1
                continue
            futures.append(
                process_and_save_one.remote(
                    row, rid, wfdb_root, prefix, gender_field,
                    DATA_DIR, args.compute_beat, args.compute_fiducial,
                )
            )

        skip_count += batch_existing
        pbar.update(batch_existing)

        for f in futures:
            result = ray.get(f)
            pbar.update(1)
            if result is not None:
                all_csv_rows.append(result["csv_row"])
                all_fn_rows.append(result["fn_row"])
                save_count += 1

    pbar.close()
    logging.info(f"  {name} 완료: {save_count:,}개 저장, {skip_count:,}개 기존, {total-save_count-skip_count:,}개 스킵")

    df_combined = df.copy()
    if "SexDSC" in df_combined.columns and "Sex" not in df_combined.columns:
        df_combined["Sex"] = df_combined["SexDSC"]
    df_combined["source"] = name
    keep_cols = [c for c in COMBINED_COLUMNS if c in df_combined.columns]
    df_combined = df_combined[keep_cols]

    return all_csv_rows, all_fn_rows, df_combined


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="HEEDB H5 변환")
    parser.add_argument("--num_cpus",       type=int, default=64)
    parser.add_argument("--batch_size",     type=int, default=5000)
    parser.add_argument("--compute_beat",    action="store_true", help="beat_annotation 생성")
    parser.add_argument("--compute_fiducial", action="store_true", help="fiducial_point/feature 생성")
    # 신호 품질은 기본 비활성화 — 필요 시 append_signal_quality.py 사용
    parser.add_argument("--compute_quality", action="store_true",
                        help="신호 품질 계산 (기본 OFF; 별도로 append_signal_quality.py 권장)")
    args = parser.parse_args()

    log_path = os.path.join(OUTPUT_ROOT, "conversion.log")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    if args.compute_quality:
        logging.warning("--compute_quality 는 변환 속도를 크게 낮춥니다. append_signal_quality.py 사용을 권장합니다.")
    logging.info(f"출력: {OUTPUT_ROOT}")
    logging.info(f"옵션: beat={args.compute_beat}, fiducial={args.compute_fiducial}, quality={args.compute_quality}")

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)
    logging.info(f"Ray CPUs: {ray.available_resources().get('CPU', 'N/A')}")

    all_csv, all_fn, all_combined = [], [], []

    for inst in INSTITUTIONS:
        csv_rows, fn_rows, combined_df = process_institution(inst, args)
        all_csv.extend(csv_rows)
        all_fn.extend(fn_rows)
        all_combined.append(combined_df)

    ray.shutdown()

    logging.info("CSV 저장 중...")

    table_path = os.path.join(OUTPUT_ROOT, "heedb_table.csv")
    pd.DataFrame(all_csv).to_csv(table_path, index=False)
    logging.info(f"  heedb_table.csv: {len(all_csv):,}행")

    fn_path = os.path.join(OUTPUT_ROOT, "file_name.csv")
    pd.DataFrame(all_fn).to_csv(fn_path, index=False)
    logging.info(f"  file_name.csv: {len(all_fn):,}행")

    combined_path = os.path.join(OUTPUT_ROOT, "combined_metadata.csv")
    combined = pd.concat(all_combined, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    logging.info(f"  combined_metadata.csv: {len(combined):,}행")

    logging.info(f"전체 완료: H5 {len(all_csv):,}개, CSV 3개")
    logging.info(f"신호 품질 계산: python append_signal_quality.py --csv {table_path} --h5_root {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()