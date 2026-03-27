"""
신호 품질 계산 → CSV 추가 스크립트
=====================================
H5 파일에서 신호를 읽어 signal_statistics + beat_similarity를 계산하고
기존 CSV에 7개 컬럼을 추가(또는 갱신)합니다.

추가 컬럼:
  nan_ratio, amp_mean, amp_std, amp_skewness, amp_kurtosis  (per-lead list, 12개)
  bs_corr, bs_dtw                                            (per-lead list, 12개)

실행:
  # HEEDB
  python append_signal_quality.py \\
      --csv  /data/h5/heedb/v4.0/heedb_table.csv \\
      --h5_root /data/h5/heedb/v4.0

  # Public (전체 테이블)
  python append_signal_quality.py \\
      --csv  /data/h5/public/v1.0/public_ecg_table.csv \\
      --h5_root /data/h5/public/v1.0

  # 특정 데이터셋만 (--dataset 필터)
  python append_signal_quality.py \\
      --csv  /data/h5/public/v1.0/public_ecg_table.csv \\
      --h5_root /data/h5/public/v1.0 \\
      --dataset ptb_xl,mimic

  # 이미 계산된 행 재계산 포함
  python append_signal_quality.py \\
      --csv  /data/h5/heedb/v4.0/heedb_table.csv \\
      --h5_root /data/h5/heedb/v4.0 \\
      --overwrite

옵션:
  --no_dtw      DTW 계산 생략 (속도 우선)
  --num_cpus    Ray 병렬 CPU 수 (기본 32)
  --batch_size  배치 크기 (기본 1000)
  --backup      덮어쓰기 전 원본 CSV 백업
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

from utils_heedb import signal_statistics, beat_similarity

# 품질 컬럼 정의
QUALITY_COLS = ["nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis", "bs_corr", "bs_dtw"]


# ═══════════════════════════════════════════════════════════════
# H5에서 신호 로드
# ═══════════════════════════════════════════════════════════════
def load_signal_from_h5(h5_path: str) -> tuple[np.ndarray, int]:
    """
    H5 파일의 첫 번째 세그먼트 신호를 읽어 반환.

    Returns:
        sig: (timepoints, 12) float32
        fs:  sampling rate (int)
    """
    with h5py.File(h5_path, "r") as f:
        fs = int(f["ECG/metadata"].attrs.get("fs", 500))
        sig = f["ECG/segments/0/signal"][()].astype(np.float32)  # (12, T)
    return sig.T, fs   # (T, 12)


# ═══════════════════════════════════════════════════════════════
# Ray remote: 단일 행 품질 계산
# ═══════════════════════════════════════════════════════════════
@ray.remote
def compute_quality_one(
    row_idx: int,
    h5_path: str,
    heedb_dir: str,
    compute_dtw: bool,
) -> dict | None:
    """
    H5 신호 로드 → signal_statistics + beat_similarity 계산.

    Returns:
        {"row_idx": int, "nan_ratio": ..., ..., "bs_corr": ..., "bs_dtw": ...}
        실패 시 None
    """
    import sys, os
    sys.path.insert(0, heedb_dir)
    from utils_heedb import signal_statistics, beat_similarity

    if not os.path.isfile(h5_path):
        return None

    try:
        import h5py, numpy as np
        with h5py.File(h5_path, "r") as f:
            fs  = int(f["ECG/metadata"].attrs.get("fs", 500))
            sig = f["ECG/segments/0/signal"][()].astype(np.float32).T  # (T, 12)
        sig = np.nan_to_num(sig)
    except Exception:
        return None

    try:
        stats = signal_statistics(sig)
    except Exception:
        stats = {k: [float("nan")] * 12 for k in
                 ["nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis"]}

    if compute_dtw:
        try:
            bs = beat_similarity(sig, sampling_rate=fs)
        except Exception:
            bs = {"bs_corr": [float("nan")] * 12, "bs_dtw": [float("nan")] * 12}
    else:
        # DTW 생략 — corr만 계산
        try:
            import neurokit2 as nk
            n_leads = sig.shape[1]
            fixed_length = fs * 2
            mean_corrs = [float("nan")] * n_leads

            for idx in range(n_leads):
                try:
                    _, rpeaks = nk.ecg_peaks(sig[:, idx], sampling_rate=fs)
                    rpeaks = rpeaks.get("ECG_R_Peaks", [])
                    if len(rpeaks) <= 3:
                        continue
                    beats = nk.ecg_segment(sig[:, idx], rpeaks, sampling_rate=fs)
                    if len(beats) <= 3:
                        continue
                    beat_matrix = []
                    for beat in beats.values():
                        br = nk.signal_resample(np.array(beat, dtype=float), desired_length=fixed_length)
                        std = np.std(br)
                        br = (br - np.mean(br)) / std if std > 0 else np.zeros_like(br)
                        beat_matrix.append(br.squeeze())
                    beat_matrix = np.array(beat_matrix)
                    corrs = [
                        np.corrcoef(beat_matrix[i], beat_matrix[i + 1])[0, 1]
                        for i in range(len(beat_matrix) - 1)
                        if not (np.any(np.isnan(beat_matrix[i])) or np.any(np.isnan(beat_matrix[i + 1])))
                    ]
                    corrs = [c for c in corrs if not np.isnan(c)]
                    mean_corrs[idx] = float(np.mean(corrs)) if corrs else float("nan")
                except Exception:
                    continue

            bs = {"bs_corr": mean_corrs, "bs_dtw": [float("nan")] * n_leads}
        except Exception:
            bs = {"bs_corr": [float("nan")] * 12, "bs_dtw": [float("nan")] * 12}

    return {
        "row_idx":       row_idx,
        "nan_ratio":     str(stats.get("nan_ratio",     [float("nan")] * 12)),
        "amp_mean":      str(stats.get("amp_mean",      [float("nan")] * 12)),
        "amp_std":       str(stats.get("amp_std",       [float("nan")] * 12)),
        "amp_skewness":  str(stats.get("amp_skewness",  [float("nan")] * 12)),
        "amp_kurtosis":  str(stats.get("amp_kurtosis",  [float("nan")] * 12)),
        "bs_corr":       str(bs.get("bs_corr",          [float("nan")] * 12)),
        "bs_dtw":        str(bs.get("bs_dtw",           [float("nan")] * 12)),
    }


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
    logging.info(f"  전체: {len(df):,}행, columns: {list(df.columns)}")

    # ── 품질 컬럼 초기화 ──────────────────────────────────────
    for col in QUALITY_COLS:
        if col not in df.columns:
            df[col] = ""

    # ── dataset 필터 ──────────────────────────────────────────
    if args.dataset:
        target_ds = [d.strip() for d in args.dataset.split(",")]
        if "dataset" in df.columns:
            mask = df["dataset"].isin(target_ds)
        else:
            # HEEDB 테이블은 dataset 컬럼이 없을 수 있음 — filepath로 판단
            mask = df["filepath"].apply(
                lambda p: any(f"/{ds}/" in str(p) or str(p).startswith(f"data/{ds}/") for ds in target_ds)
            )
        logging.info(f"  dataset 필터: {target_ds} → {mask.sum():,}행")
    else:
        mask = pd.Series([True] * len(df), index=df.index)

    # ── 계산 대상 결정 ────────────────────────────────────────
    if args.overwrite:
        target_mask = mask
    else:
        # nan_ratio 컬럼이 비어있거나 ""인 행만 처리
        not_done = df["nan_ratio"].apply(lambda v: str(v).strip() in ("", "nan", "NaN"))
        target_mask = mask & not_done

    target_idx = df.index[target_mask].tolist()
    logging.info(f"  계산 대상: {len(target_idx):,}행 ({'overwrite' if args.overwrite else '미계산 행만'})")

    if len(target_idx) == 0:
        logging.info("계산할 행이 없습니다.")
        return

    # ── 백업 ──────────────────────────────────────────────────
    if args.backup:
        backup_path = csv_path.with_suffix(".backup.csv")
        df.to_csv(backup_path, index=False)
        logging.info(f"  백업: {backup_path}")

    # ── Ray 초기화 ────────────────────────────────────────────
    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)
    logging.info(f"Ray CPUs: {ray.available_resources().get('CPU', 'N/A')}")

    # ── 배치 처리 ─────────────────────────────────────────────
    batch_size = args.batch_size
    n_batches  = (len(target_idx) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(target_idx), desc="신호 품질 계산", unit="rec")

    results_map: dict[int, dict] = {}   # row_idx → result dict
    done = 0
    fail = 0

    for b in range(n_batches):
        batch_idx = target_idx[b * batch_size : (b + 1) * batch_size]

        futures = []
        for row_idx in batch_idx:
            filepath = str(df.at[row_idx, "filepath"])
            # filepath: "data/ptb_xl/px12345_0.h5" 또는 "data/he1...h5"
            h5_path = str(h5_root / filepath)
            futures.append(
                compute_quality_one.remote(row_idx, h5_path, HEEDB_DIR, not args.no_dtw)
            )

        for fut in futures:
            result = ray.get(fut)
            pbar.update(1)
            if result is not None:
                results_map[result["row_idx"]] = result
                done += 1
            else:
                fail += 1

        # ── 배치 완료 시 중간 저장 ────────────────────────────
        if args.save_interval > 0 and (b + 1) % args.save_interval == 0:
            _apply_results(df, results_map)
            df.to_csv(csv_path, index=False)
            logging.info(f"  중간 저장 (배치 {b+1}/{n_batches}): {csv_path}")

    pbar.close()
    ray.shutdown()

    # ── 최종 적용 및 저장 ─────────────────────────────────────
    _apply_results(df, results_map)
    df.to_csv(csv_path, index=False)

    logging.info(f"\n완료: {done:,}개 성공, {fail:,}개 실패")
    logging.info(f"저장: {csv_path}")

    # ── 요약 통계 출력 ────────────────────────────────────────
    computed_rows = df.index[df["nan_ratio"].apply(lambda v: str(v).strip() not in ("", "nan", "NaN"))]
    logging.info(f"전체 품질 계산 완료 행: {len(computed_rows):,}/{len(df):,}")


def _apply_results(df: pd.DataFrame, results_map: dict):
    """results_map을 df에 반영"""
    for row_idx, result in results_map.items():
        for col in QUALITY_COLS:
            df.at[row_idx, col] = result.get(col, "")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="신호 품질 계산 후 CSV에 컬럼 추가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # HEEDB 전체
  python append_signal_quality.py \\
      --csv /data/h5/heedb/v4.0/heedb_table.csv \\
      --h5_root /data/h5/heedb/v4.0

  # Public 특정 데이터셋만
  python append_signal_quality.py \\
      --csv /data/h5/public/v1.0/public_ecg_table.csv \\
      --h5_root /data/h5/public/v1.0 \\
      --dataset ptb_xl,sph

  # DTW 생략으로 빠르게
  python append_signal_quality.py \\
      --csv /data/h5/public/v1.0/public_ecg_table.csv \\
      --h5_root /data/h5/public/v1.0 \\
      --no_dtw --num_cpus 64
        """,
    )
    parser.add_argument("--csv",       type=str, required=True,
                        help="대상 CSV 파일 경로 (heedb_table.csv 또는 public_ecg_table.csv)")
    parser.add_argument("--h5_root",   type=str, required=True,
                        help="H5 루트 경로. filepath 컬럼의 상대경로가 이 경로에 결합됩니다.")
    parser.add_argument("--dataset",   type=str, default=None,
                        help="특정 데이터셋만 처리 (쉼표 구분, 예: ptb_xl,mimic). 기본: 전체")
    parser.add_argument("--overwrite", action="store_true",
                        help="이미 계산된 행도 재계산")
    parser.add_argument("--no_dtw",   action="store_true",
                        help="DTW 계산 생략 (bs_dtw = NaN, 속도 우선)")
    parser.add_argument("--num_cpus",  type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Ray 배치 크기 (기본 1000)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="N 배치마다 중간 저장 (0이면 최종 1회만, 기본 10)")
    parser.add_argument("--backup",    action="store_true",
                        help="처리 전 원본 CSV를 .backup.csv로 백업")
    args = parser.parse_args()

    log_path = Path(args.csv).parent / "append_signal_quality.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"CSV:     {args.csv}")
    logging.info(f"H5 루트: {args.h5_root}")
    logging.info(f"옵션:    dataset={args.dataset}, overwrite={args.overwrite}, no_dtw={args.no_dtw}, num_cpus={args.num_cpus}")

    run(args)


if __name__ == "__main__":
    main()