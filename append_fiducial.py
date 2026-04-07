"""
Fiducial 계산 → 기존 H5에 추가 스크립트
=========================================
이미 변환된 H5 파일에 beat_annotation / fiducial_point / fiducial_feature를
읽어서 추가(또는 덮어쓰기)합니다.

추가 그룹 (ECG/segments/0/ 하위):
  beat_annotation/   sample, symbol, subtype, chan, num, aux_note
  fiducial_point/    fsample, fiducial
  fiducial_feature/  p_amp, q_amp, ... (19개 attrs)

실행:
  # HEEDB 전체
  python append_fiducial.py \\
      --csv     /home/irteam/opendata1/h5/heedb/v4.0/heedb_table.csv \\
      --h5_root /home/irteam/opendata1/h5/heedb/v4.0

  # physionet 전체
  python append_fiducial.py \\
      --csv     /home/irteam/opendata1/h5/physionet/v2.0/public_ecg_table.csv \\
      --h5_root /home/irteam/opendata1/h5/physionet/v2.0

  # 특정 데이터셋만
  python append_fiducial.py \\
      --csv     /home/irteam/opendata1/h5/physionet/v2.0/public_ecg_table.csv \\
      --h5_root /home/irteam/opendata1/h5/physionet/v2.0 \\
      --dataset georgia,ptbxl

  # 이미 계산된 파일도 재계산
  python append_fiducial.py ... --overwrite

  # beat_annotation만 (fiducial 제외)
  python append_fiducial.py ... --no_fiducial

옵션:
  --no_fiducial   fiducial_point / fiducial_feature 생략 (beat_annotation만)
  --no_beat       beat_annotation 생략 (fiducial만)
  --overwrite     이미 존재하는 그룹도 재계산
  --num_cpus      Ray 병렬 CPU 수 (기본 32)
  --batch_size    배치 크기 (기본 500)
  --save_interval N 배치마다 진행 로그 출력 (기본 10)
  --backup        처리 전 원본 CSV를 .backup.csv로 백업
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

SCRIPT_DIR = str(Path(__file__).resolve().parent)


# ═══════════════════════════════════════════════════════════════
# Ray remote: 단일 H5 파일에 fiducial 추가
# ═══════════════════════════════════════════════════════════════
@ray.remote
def append_fiducial_one(
    h5_path:       str,
    script_dir:    str,
    compute_beat:  bool,
    compute_fidu:  bool,
    overwrite:     bool,
) -> dict:
    """
    H5 신호 로드 → beat_annotation / fiducial_point / fiducial_feature 계산 후 H5에 기록.
    Returns: {"path": str, "status": "ok"|"skip"|"fail", "reason": str}
    """
    import sys, os
    import numpy as np
    import h5py
    sys.path.insert(0, script_dir)
    from utils.h5_structure import TARGET_SIG_NAME
    from utils.signal_processing import extract_beat_annotation, extract_fiducial

    UTF8 = h5py.string_dtype(encoding="utf-8")

    FIDUCIAL_FEATURE_KEYS = [
        "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
        "p_dur", "pr_seg", "qrs_dur", "st_seg", "t_dur",
        "pr_int", "qt_int", "rr_int", "tp_seg",
        "qtc_baz", "qtc_frid", "p_axis", "r_axis", "t_axis",
    ]

    if not os.path.isfile(h5_path):
        return {"path": h5_path, "status": "fail", "reason": "파일 없음"}

    # ── 세그먼트 목록 + skip 판단 ─────────────────────────────
    try:
        with h5py.File(h5_path, "r") as f:
            seg_grp = f.get("ECG/segments")
            if seg_grp is None:
                return {"path": h5_path, "status": "fail", "reason": "ECG/segments 없음"}
            n_segs = int(seg_grp.attrs.get("seg_len", 0))
            if n_segs == 0:
                return {"path": h5_path, "status": "fail", "reason": "seg_len=0"}
            fs = int(f["ECG/metadata"].attrs.get("fs", 500))

            if not overwrite:
                all_done = True
                for i in range(n_segs):
                    si = str(i)
                    if si not in seg_grp:
                        all_done = False
                        break
                    s = seg_grp[si]
                    if compute_beat and "beat_annotation" not in s:
                        all_done = False; break
                    if compute_fidu and "fiducial_point" not in s:
                        all_done = False; break
                if all_done:
                    return {"path": h5_path, "status": "skip", "reason": "이미 존재"}

            # 신호 로드 (전 세그먼트)
            seg_signals = []
            for i in range(n_segs):
                seg_signals.append(
                    seg_grp[str(i)]["signal"][()].astype(np.float32)
                )
    except Exception as e:
        return {"path": h5_path, "status": "fail", "reason": f"H5 읽기 실패: {e}"}

    # ── 세그먼트별 beat / fiducial 계산 ───────────────────────
    ba_list = fp_list = ff_list = None
    if compute_beat:
        ba_list = []
        for sig in seg_signals:
            try:
                ba_list.append(extract_beat_annotation(np.nan_to_num(sig[1]), fs))
            except Exception:
                ba_list.append({"sample": [], "symbol": [], "subtype": [],
                                "chan": [], "num": [], "aux_note": []})
    if compute_fidu:
        fp_list, ff_list = [], []
        for sig in seg_signals:
            try:
                fp, ff = extract_fiducial(np.nan_to_num(sig), fs)
            except Exception:
                fp = {"fsample": [], "fiducial": []}
                ff = {k: np.float16(np.nan) for k in FIDUCIAL_FEATURE_KEYS}
            fp_list.append(fp); ff_list.append(ff)

    # ── H5에 기록 ─────────────────────────────────────────────
    try:
        with h5py.File(h5_path, "a") as f:
            seg_grp = f["ECG/segments"]
            for i in range(n_segs):
                s = seg_grp[str(i)]

                if ba_list is not None:
                    if "beat_annotation" in s:
                        del s["beat_annotation"]
                    ba      = ba_list[i]
                    ba_grp  = s.create_group("beat_annotation")
                    samples = np.array(ba.get("sample", []), dtype=np.int16)
                    nb      = len(samples)
                    ba_grp.create_dataset("sample",   data=samples)
                    ba_grp.create_dataset("symbol",   data=np.array(ba.get("symbol",   [""]*nb), dtype=UTF8), dtype=UTF8)
                    ba_grp.create_dataset("subtype",  data=np.array(ba.get("subtype",  np.zeros(nb)), dtype=np.int16))
                    ba_grp.create_dataset("chan",     data=np.array(ba.get("chan",     np.zeros(nb)), dtype=np.int16))
                    ba_grp.create_dataset("num",      data=np.array(ba.get("num",      np.zeros(nb)), dtype=np.int16))
                    ba_grp.create_dataset("aux_note", data=np.array(ba.get("aux_note", [""]*nb), dtype=UTF8), dtype=UTF8)

                if fp_list is not None:
                    if "fiducial_point" in s:
                        del s["fiducial_point"]
                    fp      = fp_list[i]
                    fp_grp  = s.create_group("fiducial_point")
                    fs_arr  = fp.get("fsample",  [])
                    fid_arr = fp.get("fiducial", [])
                    fp_grp.create_dataset(
                        "fsample",
                        data=np.array(fs_arr,  dtype=np.int16) if len(fs_arr)  else np.array([], dtype=np.int16),
                    )
                    fp_grp.create_dataset(
                        "fiducial",
                        data=np.array(fid_arr, dtype=UTF8)    if len(fid_arr) else np.array([], dtype=UTF8),
                        dtype=UTF8,
                    )

                if ff_list is not None:
                    if "fiducial_feature" in s:
                        del s["fiducial_feature"]
                    ff     = ff_list[i]
                    ff_grp = s.create_group("fiducial_feature")
                    for key in FIDUCIAL_FEATURE_KEYS:
                        val = ff.get(key, np.nan)
                        try:
                            ff_grp.attrs[key] = np.float16(val)
                        except (TypeError, ValueError):
                            ff_grp.attrs[key] = np.float16(np.nan)

            if ba_list is not None:
                f.attrs["beat_ext_method"] = "neurokit2"
            if ff_list is not None:
                f.attrs["fidu_extract_method"] = "neurokit2-dwt"

    except Exception as e:
        return {"path": h5_path, "status": "fail", "reason": f"H5 쓰기 실패: {e}"}

    return {"path": h5_path, "status": "ok", "reason": ""}


# ═══════════════════════════════════════════════════════════════
# 메인 처리
# ═══════════════════════════════════════════════════════════════
def run(args):
    csv_path = Path(args.csv)
    h5_root  = Path(args.h5_root)

    if not csv_path.exists():
        logging.error(f"CSV 파일 없음: {csv_path}")
        return

    logging.info(f"CSV 로드: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    logging.info(f"  전체: {len(df):,}행")

    compute_beat = not args.no_beat
    compute_fidu = not args.no_fiducial

    if not compute_beat and not compute_fidu:
        logging.error("--no_beat 와 --no_fiducial 동시 사용 불가")
        return

    logging.info(f"  계산 항목: beat={compute_beat}, fiducial={compute_fidu}, overwrite={args.overwrite}")

    # ── dataset 필터 ──────────────────────────────────────────
    if args.dataset:
        target_ds = [d.strip() for d in args.dataset.split(",")]
        if "dataset" in df.columns:
            mask = df["dataset"].isin(target_ds)
        else:
            mask = df["filepath"].apply(
                lambda p: any(f"/{ds}/" in str(p) or f"/{ds}." in str(p) for ds in target_ds)
            )
        logging.info(f"  dataset 필터: {target_ds} → {mask.sum():,}행")
        df = df[mask].reset_index(drop=True)

    if len(df) == 0:
        logging.warning("대상 행 없음")
        return

    # ── 백업 ──────────────────────────────────────────────────
    if args.backup:
        backup_path = csv_path.with_suffix(".backup.csv")
        pd.read_csv(csv_path, dtype=str).to_csv(backup_path, index=False)
        logging.info(f"  백업: {backup_path}")

    # ── H5 경로 목록 ──────────────────────────────────────────
    h5_paths = [str(h5_root / row["filepath"]) for _, row in df.iterrows()]
    total    = len(h5_paths)
    logging.info(f"  처리 대상: {total:,}개")

    # ── Ray 초기화 ────────────────────────────────────────────
    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)
    logging.info(f"Ray CPUs: {ray.available_resources().get('CPU', 'N/A')}")

    # ── 배치 처리 ─────────────────────────────────────────────
    batch_size = args.batch_size
    n_batches  = (total + batch_size - 1) // batch_size
    pbar = tqdm(total=total, desc="fiducial 추가", unit="rec")

    ok = skip = fail = 0

    for b in range(n_batches):
        batch = h5_paths[b * batch_size : (b + 1) * batch_size]
        futures = [
            append_fiducial_one.remote(
                p, SCRIPT_DIR, compute_beat, compute_fidu, args.overwrite,
            )
            for p in batch
        ]
        for res in ray.get(futures):
            pbar.update(1)
            s = res["status"]
            if   s == "ok":   ok   += 1
            elif s == "skip": skip += 1
            else:
                fail += 1
                logging.debug(f"  FAIL: {res['path']} — {res['reason']}")

        if args.save_interval > 0 and (b + 1) % args.save_interval == 0:
            logging.info(f"  진행 [{b+1}/{n_batches}] ok={ok:,} skip={skip:,} fail={fail:,}")

    pbar.close()
    ray.shutdown()

    logging.info(f"\n완료: ok={ok:,} | skip={skip:,} | fail={fail:,} / 전체={total:,}")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="기존 H5에 beat_annotation / fiducial 추가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # physionet 전체
  python append_fiducial.py \\
      --csv     /home/irteam/opendata1/h5/physionet/v2.0/public_ecg_table.csv \\
      --h5_root /home/irteam/opendata1/h5/physionet/v2.0

  # 특정 데이터셋만
  python append_fiducial.py \\
      --csv     /home/irteam/opendata1/h5/physionet/v2.0/public_ecg_table.csv \\
      --h5_root /home/irteam/opendata1/h5/physionet/v2.0 \\
      --dataset georgia,ptbxl --num_cpus 64

  # beat만 (fiducial 제외)
  python append_fiducial.py ... --no_fiducial

  # 재계산 강제
  python append_fiducial.py ... --overwrite
        """,
    )
    parser.add_argument("--csv",          type=str, required=True)
    parser.add_argument("--h5_root",      type=str, required=True)
    parser.add_argument("--dataset",      type=str, default=None,
                        help="쉼표 구분: georgia,ptbxl")
    parser.add_argument("--overwrite",    action="store_true",
                        help="이미 존재하는 그룹도 재계산")
    parser.add_argument("--no_beat",      action="store_true",
                        help="beat_annotation 생략")
    parser.add_argument("--no_fiducial",  action="store_true",
                        help="fiducial_point / fiducial_feature 생략")
    parser.add_argument("--num_cpus",     type=int, default=32)
    parser.add_argument("--batch_size",   type=int, default=500)
    parser.add_argument("--save_interval",type=int, default=10,
                        help="N 배치마다 진행 로그 출력")
    parser.add_argument("--backup",       action="store_true")
    args = parser.parse_args()

    log_path = Path(args.csv).parent / "append_fiducial.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"CSV     : {args.csv}")
    logging.info(f"H5 루트 : {args.h5_root}")
    logging.info(f"옵션    : dataset={args.dataset}, overwrite={args.overwrite}, "
                 f"beat={not args.no_beat}, fiducial={not args.no_fiducial}, "
                 f"num_cpus={args.num_cpus}")

    run(args)


if __name__ == "__main__":
    main()