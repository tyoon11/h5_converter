"""
Public ECG Dataset → H5 변환 파이프라인
=========================================
지원 데이터셋:
  PTB-XL (500Hz), PTB (1000Hz), SPH (500Hz), EchoNext (250Hz),
  ZZU-pECG (500Hz), CODE-15% (400Hz), Chapman (500Hz),
  CPSC2018 (500Hz), CPSC-Extra (500Hz), Georgia (500Hz),
  Ningbo (500Hz), MIMIC-IV-ECG (500Hz)

실행:
  python convert_to_h5_public.py --dataset_dir /data/raw --output_root /data/h5/public/v1.0 --dataset ptb_xl
  python convert_to_h5_public.py --dataset_dir /data/raw --output_root /data/h5/public/v1.0 --dataset all --num_cpus 32
  python convert_to_h5_public.py --dataset_dir /data/raw --output_root /data/h5/public/v1.0 --dataset mimic --compute_beat

신호 품질(signal quality) 계산은 기본 비활성화입니다.
변환 완료 후 별도로 계산하세요:
  python append_signal_quality.py --csv /data/h5/public/v1.0/public_ecg_table.csv --h5_root /data/h5/public/v1.0

canonical 채널 인덱스 (clinical_ts 기준):
  {i:0, ii:1, iii:2, avr:3, avl:4, avf:5, v1:6, v2:7, v3:8, v4:9, v5:10, v6:11}
TARGET_SIG_NAME → canonical 인덱스:
  [I,  II, III, V1, V2, V3, V4, V5, V6, aVF, aVL, aVR]
  [0,  1,  2,   6,  7,  8,  9, 10, 11,  5,   4,   3 ]
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import h5py
import ray
from pathlib import Path
from tqdm import tqdm

# heedb/ 모듈 경로
HEEDB_DIR = str(Path(__file__).resolve().parent / "heedb")
sys.path.insert(0, HEEDB_DIR)

# clinical_ts import
CODE_DIR = str(Path(__file__).resolve().parent / "code")
if os.path.isdir(CODE_DIR):
    sys.path.insert(0, CODE_DIR)

try:
    from clinical_ts.utils.ecg_utils import *          # noqa: F401,F403
    from clinical_ts.data.time_series_dataset_utils import *  # noqa: F401,F403
    CLINICAL_TS_OK = True
except ImportError:
    CLINICAL_TS_OK = False
    print("[WARNING] clinical_ts import 실패. prepare 함수들을 사용할 수 없습니다.")

from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME

# ═══════════════════════════════════════════════════════════════
# 채널 reorder 상수
# ═══════════════════════════════════════════════════════════════
# canonical → TARGET 순서 인덱스
CANONICAL_TO_TARGET_IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11, 5, 4, 3]

# CODE-15% 8채널 매핑
CODE15_TARGET_POS = [0, 1, 3, 4, 5, 6, 7, 8]   # TARGET_SIG_NAME 내 위치 (I,II,V1-V6)

CHANNEL_ITOS = "canonical"


# ═══════════════════════════════════════════════════════════════
# age / gender 헬퍼
# ═══════════════════════════════════════════════════════════════
def _default_age(row):
    try:
        v = float(row["age"])
        if 0.0 <= v <= 1.5:
            return round(v, 6)
        return round(v / 100.0, 6)
    except (KeyError, TypeError, ValueError):
        return -1.0


def _default_gender(row):
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
# 데이터셋별 설정 레지스트리
# ═══════════════════════════════════════════════════════════════
def _build_configs(dataset_dir: Path, output_root: Path) -> dict:
    return {
        "ptb_xl": dict(
            prefix="px", fs=500, n_channels=12,
            data_dir=dataset_dir / "ptb-xl",
            target_dir=output_root / "ptb_xl" / "records500",
            prepare_fn="prepare_data_ptb_xl", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "ptb": dict(
            prefix="pt", fs=1000, n_channels=15,
            data_dir=dataset_dir / "ptb",
            target_dir=output_root / "ptb",
            prepare_fn="prepare_ptbv2", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "sph": dict(
            prefix="sp", fs=500, n_channels=12,
            data_dir=dataset_dir / "sph",
            target_dir=output_root / "sph",
            prepare_fn="prepare_data_sph", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "echonext": dict(
            prefix="en", fs=250, n_channels=12,
            data_dir=dataset_dir / "echonext",
            target_dir=output_root / "echonext",
            prepare_fn="prepare_data_echonext", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "zzu_pecg": dict(
            prefix="zz", fs=500, n_channels=12,
            data_dir=dataset_dir / "zzu_pecg",
            target_dir=output_root / "zzu_pecg",
            prepare_fn="prepare_data_zzu_pecg", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "code15": dict(
            prefix="c1", fs=400, n_channels=8,
            data_dir=dataset_dir / "code15",
            target_dir=output_root / "code15",
            prepare_fn="prepare_data_ribeiro_full", prepare_kwargs={"code15": True},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "chapman": dict(
            prefix="ch", fs=500, n_channels=12,
            data_dir=dataset_dir / "chapman",
            target_dir=output_root / "chapman",
            prepare_fn="prepare_data_chapman", prepare_kwargs={"denoised": False},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "cpsc2018": dict(
            prefix="cs", fs=500, n_channels=12,
            data_dir=dataset_dir / "cpsc2018",
            target_dir=output_root / "cpsc2018",
            prepare_fn="prepare_data_cpsc2018", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "cpsc_extra": dict(
            prefix="ce", fs=500, n_channels=12,
            data_dir=dataset_dir / "cpsc_extra",
            target_dir=output_root / "cpsc_extra",
            prepare_fn="prepare_data_cpsc_extra", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "georgia": dict(
            prefix="ge", fs=500, n_channels=12,
            data_dir=dataset_dir / "georgia",
            target_dir=output_root / "georgia",
            prepare_fn="prepare_data_georgia", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "ningbo": dict(
            prefix="nb", fs=500, n_channels=12,
            data_dir=dataset_dir / "ningbo",
            target_dir=output_root / "ningbo",
            prepare_fn="prepare_data_ningbo", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
        "mimic": dict(
            prefix="mi", fs=500, n_channels=12,
            data_dir=dataset_dir / "mimic",
            target_dir=output_root / "mimic",
            prepare_fn="prepare_mimicecg", prepare_kwargs={},
            age_fn=_default_age, gender_fn=_default_gender,
        ),
    }


# ═══════════════════════════════════════════════════════════════
# 신호 로드 & 채널 reorder
# ═══════════════════════════════════════════════════════════════
def load_signal_from_npy(npy_path: str, n_channels: int, dataset_name: str) -> np.ndarray:
    """
    clinical_ts가 생성한 .npy 로드 → (12, timepoints) float16.
    .npy shape: (n_channels, timepoints) — canonical 채널 순서
    """
    sig = np.load(npy_path)
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]
    if sig.shape[0] > sig.shape[1]:
        sig = sig.T   # (C, T) 보장

    n_ch = sig.shape[0]

    if dataset_name == "code15":
        T = sig.shape[1]
        out = np.full((12, T), np.nan, dtype=np.float32)
        for src_i, tgt_i in enumerate(CODE15_TARGET_POS):
            if src_i < n_ch:
                out[tgt_i] = sig[src_i]
        return out.astype(np.float16)

    if dataset_name == "ptb" and n_ch == 15:
        sig = sig[:12, :]

    if n_ch >= 12:
        return sig[CANONICAL_TO_TARGET_IDX, :].astype(np.float16)

    raise ValueError(f"채널 수 부족: {n_ch}")


# ═══════════════════════════════════════════════════════════════
# Ray remote: 1 레코드 → H5 저장
# ═══════════════════════════════════════════════════════════════
@ray.remote
def process_one(
    row_dict: dict,
    rid: int,
    dataset_name: str,
    prefix: str,
    fs: int,
    n_channels: int,
    h5_dir: str,
    heedb_dir: str,
    compute_beat: bool,
    compute_fiducial: bool,
):
    import sys, os, h5py, numpy as np
    sys.path.insert(0, heedb_dir)
    from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME
    from utils_heedb import extract_beat_annotation, extract_fiducial
    from convert_to_h5_public import load_signal_from_npy

    npy_path = row_dict.get("data", "")
    if not npy_path or not os.path.isfile(str(npy_path)):
        return None

    try:
        sig = load_signal_from_npy(str(npy_path), n_channels, dataset_name)
    except Exception:
        return None

    if sig.shape[0] != 12:
        return None

    pid = str(row_dict.get("patient_id", row_dict.get("pid", rid))).strip()
    file_name = f"{prefix}{pid}_{rid}"
    oid = f"{file_name}_0"

    ba_list, beat_method = None, ""
    if compute_beat:
        try:
            ba = extract_beat_annotation(np.nan_to_num(sig[1].astype(np.float32)), fs)
            ba_list = [ba]
            beat_method = "neurokit2"
        except Exception:
            pass

    fp_list, ff_list, fidu_method = None, None, ""
    if compute_fiducial:
        try:
            fp, ff = extract_fiducial(np.nan_to_num(sig.astype(np.float32)), fs)
            fp_list = [fp]
            ff_list = [ff]
            fidu_method = "neurokit2-dwt"
        except Exception:
            pass

    h5_path = os.path.join(h5_dir, f"{file_name}.h5")
    try:
        with h5py.File(h5_path, "w") as h5f:
            create_h5_structure(
                h5f,
                file_name=file_name,
                beat_ext_method=beat_method,
                fidu_extract_method=fidu_method,
                record_name=str(row_dict.get("record_name", file_name)),
                n_sig=12, fs=fs, sig_len=sig.shape[1],
                base_date=str(row_dict.get("base_date", row_dict.get("acquisition_date", ""))),
                sig_name=TARGET_SIG_NAME,
                signal=[sig], seg_len=1,
                beat_annotation=ba_list,
                fiducial_point=fp_list,
                fiducial_feature=ff_list,
            )
    except Exception:
        return None

    return {
        "filepath": f"data/{dataset_name}/{file_name}.h5",
        "dataset": dataset_name,
        "pid": pid, "rid": rid, "sid": 0, "oid": oid,
        "age": row_dict.get("_age", -1),
        "gender": row_dict.get("_gender", 0),
        "height": np.nan, "weight": np.nan,
        "fs": fs,
        "channel_name": str(TARGET_SIG_NAME),
        "source_npy": str(npy_path),
    }


# ═══════════════════════════════════════════════════════════════
# 데이터셋 1개 처리
# ═══════════════════════════════════════════════════════════════
def process_dataset(dataset_name: str, cfg: dict, output_root: Path, args) -> list:
    prefix     = cfg["prefix"]
    fs         = cfg["fs"]
    n_channels = cfg["n_channels"]
    target_dir = Path(cfg["target_dir"])
    data_dir   = Path(cfg["data_dir"])
    h5_dir     = output_root / "data" / dataset_name

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(h5_dir, exist_ok=True)

    logging.info(f"\n{'='*60}")
    logging.info(f"  {dataset_name.upper()} (fs={fs}Hz, prefix={prefix})")
    logging.info(f"{'='*60}")

    if not CLINICAL_TS_OK:
        logging.error(f"  clinical_ts 없음 — {dataset_name} 건너뜀")
        return []

    prepare_fn = globals().get(cfg["prepare_fn"])
    if prepare_fn is None:
        logging.error(f"  함수 '{cfg['prepare_fn']}' 없음 — {dataset_name} 건너뜀")
        return []

    logging.info(f"  {cfg['prepare_fn']}() 실행 중...")
    try:
        kwargs = {"data_path": data_dir, "target_folder": target_dir}
        kwargs.update(cfg.get("prepare_kwargs", {}))
        df, _, _, _ = prepare_fn(**kwargs)
    except Exception as e:
        logging.error(f"  prepare 실패: {e}")
        return []

    logging.info(f"  df: {df.shape[0]:,}행")

    df["_age"]    = df.apply(cfg["age_fn"],    axis=1)
    df["_gender"] = df.apply(cfg["gender_fn"], axis=1)

    existing = {f[:-3] for f in os.listdir(h5_dir) if f.endswith(".h5")}
    logging.info(f"  이미 변환: {len(existing):,}개")

    rows       = df.to_dict("records")
    total      = len(rows)
    batch_size = args.batch_size
    n_batches  = (total + batch_size - 1) // batch_size

    all_csv_rows = []
    pbar  = tqdm(total=total, desc=f"  {dataset_name}", unit="rec")
    saved = 0

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end   = min(start + batch_size, total)
        batch = rows[start:end]

        futures = []
        skip_n  = 0
        for offset, row in enumerate(batch):
            rid = start + offset
            pid = str(row.get("patient_id", row.get("pid", rid))).strip()
            fn  = f"{prefix}{pid}_{rid}"
            if fn in existing:
                skip_n += 1
                continue
            futures.append(
                process_one.remote(
                    row, rid, dataset_name, prefix, fs, n_channels,
                    str(h5_dir), HEEDB_DIR,
                    args.compute_beat, args.compute_fiducial,
                )
            )

        pbar.update(skip_n)
        for fut in futures:
            result = ray.get(fut)
            pbar.update(1)
            if result is not None:
                all_csv_rows.append(result)
                saved += 1

    pbar.close()
    logging.info(f"  완료: {saved:,}개 저장, {total-saved:,}개 스킵/실패")
    return all_csv_rows


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Public ECG H5 변환")
    parser.add_argument("--dataset_dir",  type=str, required=True,
                        help="원본 데이터셋 루트 (예: /data/ecg_datasets)")
    parser.add_argument("--output_root",  type=str, required=True,
                        help="H5 출력 루트 (예: /data/h5/public/v1.0)")
    parser.add_argument("--dataset",      type=str, default="all",
                        help="all / 쉼표 구분: ptb_xl,mimic / 단일: ptb_xl")
    parser.add_argument("--num_cpus",     type=int, default=32)
    parser.add_argument("--batch_size",   type=int, default=2000)
    parser.add_argument("--compute_beat",     action="store_true", help="beat_annotation 생성")
    parser.add_argument("--compute_fiducial", action="store_true", help="fiducial_point/feature 생성")
    # 신호 품질은 기본 비활성화 — append_signal_quality.py 로 별도 계산 권장
    parser.add_argument("--compute_quality", action="store_true",
                        help="신호 품질 계산 (기본 OFF; append_signal_quality.py 권장)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
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
    if args.compute_quality:
        logging.warning("--compute_quality 는 변환 속도를 크게 낮춥니다. append_signal_quality.py 사용을 권장합니다.")
    logging.info(f"출력: {output_root}")
    logging.info(f"옵션: beat={args.compute_beat}, fiducial={args.compute_fiducial}")

    configs = _build_configs(dataset_dir, output_root)

    if args.dataset.lower() == "all":
        target_datasets = list(configs.keys())
    else:
        target_datasets = [d.strip() for d in args.dataset.split(",")]
        unknown = [d for d in target_datasets if d not in configs]
        if unknown:
            logging.error(f"알 수 없는 데이터셋: {unknown}  선택 가능: {list(configs.keys())}")
            return

    logging.info(f"변환 대상: {target_datasets}")

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)
    logging.info(f"Ray CPUs: {ray.available_resources().get('CPU', 'N/A')}")

    all_csv = []
    for ds_name in target_datasets:
        rows = process_dataset(ds_name, configs[ds_name], output_root, args)
        all_csv.extend(rows)

    ray.shutdown()

    if all_csv:
        table_path = output_root / "public_ecg_table.csv"
        df_all = pd.DataFrame(all_csv)
        df_all.to_csv(table_path, index=False)
        logging.info(f"\npublic_ecg_table.csv: {len(all_csv):,}행 → {table_path}")

        for ds_name in df_all["dataset"].unique():
            sub_path = output_root / f"{ds_name}_table.csv"
            df_all[df_all["dataset"] == ds_name].to_csv(sub_path, index=False)
            logging.info(f"  {ds_name}_table.csv: {len(df_all[df_all['dataset']==ds_name]):,}행")

        logging.info(f"\n신호 품질 계산:")
        logging.info(f"  python append_signal_quality.py --csv {table_path} --h5_root {output_root}")
    else:
        logging.warning("변환된 레코드가 없습니다.")

    logging.info("전체 완료")


if __name__ == "__main__":
    main()