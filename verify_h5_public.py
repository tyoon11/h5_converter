"""
공개 데이터셋 H5 검증 스크립트
================================
단일 파일 상세 검증 + 폴더/데이터셋 일괄 검증

실행:
  # 단일 파일 상세 검증
  python verify_h5_public.py --file /data/h5/public/v1.0/data/ptb_xl/px12345_0.h5

  # 특정 데이터셋 폴더 검증
  python verify_h5_public.py --dir /data/h5/public/v1.0/data/ptb_xl --sample 200

  # output_root 아래 전체 데이터셋 일괄 검증
  python verify_h5_public.py --output_root /data/h5/public/v1.0 --sample 200

  # 특정 데이터셋만 선택
  python verify_h5_public.py --output_root /data/h5/public/v1.0 \\
      --dataset ptb_xl,mimic --sample 100

  # CODE-15% 등 NaN 리드 허용
  python verify_h5_public.py --dir /data/h5/public/v1.0/data/code15 \\
      --allow_nan_leads
"""

import os
import sys
import argparse
import h5py
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']

FIDUCIAL_FEATURE_KEYS = [
    "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
    "p_dur", "pr_seg", "qrs_dur", "st_seg", "t_dur",
    "pr_int", "qt_int", "rr_int", "tp_seg",
    "qtc_baz", "qtc_frid", "p_axis", "r_axis", "t_axis",
]

# NaN 리드가 일부 허용되는 데이터셋 (8채널)
PARTIAL_LEAD_DATASETS = {"code15"}


# ═══════════════════════════════════════════════════════════════
# 단일 파일 상세 출력
# ═══════════════════════════════════════════════════════════════
def inspect_one(h5_path: str):
    print(f"\n{'='*60}")
    print(f"  {h5_path}")
    print(f"{'='*60}\n")

    with h5py.File(h5_path, "r") as f:
        # root attrs
        print("[root attrs]")
        for k in ["dataset_version", "file_name", "beat_ext_method", "fidu_extract_method"]:
            print(f"  {k}: {f.attrs.get(k, '(missing)')}")

        # metadata
        if "ECG" not in f or "metadata" not in f["ECG"]:
            print("\n[ERROR] ECG/metadata 없음")
            return
        meta = f["ECG/metadata"]
        print("\n[ECG/metadata attrs]")
        for k in ["record_name", "n_sig", "fs", "sig_len", "base_time", "base_date", "dtype"]:
            print(f"  {k}: {meta.attrs.get(k, '(missing)')}")
        if "sig_name" in meta:
            sn = [s.decode() if isinstance(s, bytes) else s for s in meta["sig_name"][()]]
            print(f"  sig_name     : {sn}")
            print(f"  sig_name 일치: {sn == TARGET_SIG_NAME}")

        # segments
        if "segments" not in f["ECG"]:
            print("\n[ERROR] ECG/segments 없음")
            return
        segs    = f["ECG/segments"]
        seg_len = segs.attrs.get("seg_len", 0)
        print(f"\n[ECG/segments] seg_len={seg_len}")

        for i in range(seg_len):
            si = str(i)
            if si not in segs:
                print(f"  segment {i}: 누락")
                continue
            sg = segs[si]
            print(f"\n  ── segment {i} ──")

            # signal
            if "signal" in sg:
                sig     = sg["signal"][()].astype(np.float32)
                ds      = sg["signal"]
                nan_per_lead  = np.sum(np.isnan(sig), axis=1)
                full_nan_cnt  = int(np.sum(np.all(np.isnan(sig), axis=1)))
                print(f"  signal: shape={ds.shape}  dtype={ds.dtype}")
                print(f"  전체 NaN 리드: {full_nan_cnt}/12")
                if full_nan_cnt < 12:
                    print(f"  min={np.nanmin(sig):.4f}  max={np.nanmax(sig):.4f}"
                          f"  mean={np.nanmean(sig):.4f}")
                print(f"  리드별 NaN 샘플 수: {nan_per_lead.tolist()}")
            else:
                print(f"  signal: (없음)")

            # beat_annotation
            if "beat_annotation" in sg:
                ba      = sg["beat_annotation"]
                n_peaks = ba["sample"].shape[0]
                print(f"  beat_annotation: {n_peaks}개 R-peak"
                      f"  first5={ba['sample'][:5].tolist()}")

            # fiducial_point
            if "fiducial_point" in sg:
                fp = sg["fiducial_point"]
                print(f"  fiducial_point: {fp['fsample'].shape[0]}개 포인트")

            # fiducial_feature
            if "fiducial_feature" in sg:
                ff      = sg["fiducial_feature"]
                vals    = {k: ff.attrs.get(k, float("nan")) for k in FIDUCIAL_FEATURE_KEYS}
                nan_cnt = sum(1 for v in vals.values()
                              if isinstance(v, (float, np.floating)) and np.isnan(v))
                print(f"  fiducial_feature: NaN {nan_cnt}/19")
                for k, v in vals.items():
                    print(f"    {k:12s}: {v}")


# ═══════════════════════════════════════════════════════════════
# 단일 파일 검증 (문제 리스트 반환)
# ═══════════════════════════════════════════════════════════════
def validate_one(h5_path: str, allow_nan_leads: bool = False) -> list[str]:
    issues = []
    try:
        with h5py.File(h5_path, "r") as f:
            # root attrs
            for k in ["dataset_version", "file_name"]:
                if k not in f.attrs:
                    issues.append(f"root attr '{k}' 누락")

            if "ECG" not in f:
                issues.append("ECG/ 그룹 없음")
                return issues

            # metadata
            if "metadata" not in f["ECG"]:
                issues.append("ECG/metadata 없음")
            else:
                meta = f["ECG/metadata"]
                for k in ["record_name", "n_sig", "fs", "sig_len"]:
                    if k not in meta.attrs:
                        issues.append(f"metadata attr '{k}' 누락")
                if "sig_name" in meta:
                    sn = [s.decode() if isinstance(s, bytes) else s
                          for s in meta["sig_name"][()]]
                    if sn != TARGET_SIG_NAME:
                        issues.append(f"sig_name 불일치: {sn}")

            # segments
            if "segments" not in f["ECG"]:
                issues.append("ECG/segments 없음")
                return issues

            segs    = f["ECG/segments"]
            seg_len = segs.attrs.get("seg_len", 0)
            if seg_len == 0:
                issues.append("seg_len = 0")

            for i in range(seg_len):
                si = str(i)
                if si not in segs:
                    issues.append(f"segment {i} 없음")
                    continue
                if "signal" not in segs[si]:
                    issues.append(f"segment {i}: signal 없음")
                else:
                    sig = segs[si]["signal"][()]
                    if sig.shape[0] != 12:
                        issues.append(f"segment {i}: signal shape[0]={sig.shape[0]} ≠ 12")
                    if not allow_nan_leads:
                        full_nan = int(np.sum(np.all(np.isnan(sig.astype(np.float32)), axis=1)))
                        if full_nan > 0:
                            issues.append(f"segment {i}: 전체 NaN 리드 {full_nan}개")
    except Exception as e:
        issues.append(f"파일 열기 실패: {e}")
    return issues


# ═══════════════════════════════════════════════════════════════
# 폴더 일괄 검증
# ═══════════════════════════════════════════════════════════════
def batch_validate(
    h5_dir: str,
    dataset_name: str = "",
    sample_n: int = None,
    allow_nan_leads: bool = False,
) -> dict:
    """
    h5_dir 내 .h5 파일 일괄 검증.
    dataset_name이 PARTIAL_LEAD_DATASETS에 속하면 NaN 리드 자동 허용.
    결과 통계 dict 반환.
    """
    files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]
    if not files:
        print(f"  [SKIP] H5 파일 없음: {h5_dir}")
        return {}

    # PARTIAL_LEAD_DATASETS이면 NaN 리드 자동 허용
    if dataset_name in PARTIAL_LEAD_DATASETS:
        allow_nan_leads = True

    if sample_n and sample_n < len(files):
        import random
        random.seed(42)
        files = random.sample(files, sample_n)

    ok         = 0
    fail_files = []
    shape_dist = Counter()
    fs_dist    = Counter()
    nan_lead_dist = Counter()

    for fname in tqdm(files, desc=f"  {dataset_name or h5_dir}", leave=False):
        path   = os.path.join(h5_dir, fname)
        issues = validate_one(path, allow_nan_leads=allow_nan_leads)
        if issues:
            fail_files.append((fname, issues))
        else:
            ok += 1
            try:
                with h5py.File(path, "r") as f:
                    fs_dist[int(f["ECG/metadata"].attrs.get("fs", 0))] += 1
                    if "0" in f["ECG/segments"]:
                        sig = f["ECG/segments/0/signal"][()].astype(np.float32)
                        shape_dist[sig.shape] += 1
                        full_nan = int(np.sum(np.all(np.isnan(sig), axis=1)))
                        nan_lead_dist[full_nan] += 1
            except Exception:
                pass

    total = len(files)
    return {
        "dataset":       dataset_name,
        "total":         total,
        "ok":            ok,
        "fail":          total - ok,
        "fail_files":    fail_files,
        "fs_dist":       dict(fs_dist),
        "shape_dist":    dict(shape_dist),
        "nan_lead_dist": dict(nan_lead_dist),
    }


def print_batch_result(result: dict, show_fail_n: int = 10):
    ds    = result["dataset"] or "unknown"
    total = result["total"]
    ok    = result["ok"]
    fail  = result["fail"]
    pct   = 100 * ok / total if total else 0

    status = "✅" if fail == 0 else "❌"
    print(f"  {status}  {ds:15s} | {ok:>7,}/{total:>7,} ({pct:5.1f}%)"
          f"  fs={result['fs_dist']}  NaN리드분포={result['nan_lead_dist']}")

    if result["fail_files"] and show_fail_n > 0:
        print(f"     문제 파일 (상위 {show_fail_n}):")
        for fname, issues in result["fail_files"][:show_fail_n]:
            print(f"       {fname}: {issues}")


# ═══════════════════════════════════════════════════════════════
# output_root 전체 검증
# ═══════════════════════════════════════════════════════════════
def validate_output_root(
    output_root: str,
    target_datasets: list[str] | None,
    sample_n: int | None,
    allow_nan_leads: bool,
):
    data_root = Path(output_root) / "data"
    if not data_root.exists():
        print(f"[ERROR] data/ 폴더 없음: {data_root}")
        return

    ds_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if target_datasets:
        ds_dirs = [d for d in ds_dirs if d.name in target_datasets]

    if not ds_dirs:
        print("[ERROR] 검증할 데이터셋 폴더가 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"  전체 검증: {output_root}")
    print(f"  대상: {[d.name for d in ds_dirs]}")
    print(f"{'='*60}\n")

    all_results  = []
    grand_total  = 0
    grand_ok     = 0

    for ds_dir in ds_dirs:
        result = batch_validate(
            str(ds_dir),
            dataset_name   = ds_dir.name,
            sample_n       = sample_n,
            allow_nan_leads= allow_nan_leads,
        )
        if not result:
            continue
        all_results.append(result)
        grand_total += result["total"]
        grand_ok    += result["ok"]

    print(f"\n{'─'*60}")
    print(f"  데이터셋별 결과")
    print(f"{'─'*60}")
    for r in all_results:
        print_batch_result(r)

    print(f"\n{'─'*60}")
    grand_pct = 100 * grand_ok / grand_total if grand_total else 0
    print(f"  전체 합계: {grand_ok:,}/{grand_total:,} ({grand_pct:.1f}%)")
    print(f"{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="공개 데이터셋 H5 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모드:
  --file         단일 파일 상세 출력 + 검증
  --dir          특정 폴더 일괄 검증
  --output_root  output_root/data/ 아래 전체 데이터셋 일괄 검증

예시:
  python verify_h5_public.py \\
      --file /data/h5/public/v1.0/data/ptb_xl/px12345_0.h5

  python verify_h5_public.py \\
      --dir /data/h5/public/v1.0/data/ptb_xl --sample 200

  python verify_h5_public.py \\
      --output_root /data/h5/public/v1.0 --sample 200

  python verify_h5_public.py \\
      --output_root /data/h5/public/v1.0 \\
      --dataset ptb_xl,mimic --sample 100
        """,
    )
    # 모드 (셋 중 하나)
    parser.add_argument("--file",        type=str, default=None,
                        help="단일 H5 파일 경로")
    parser.add_argument("--dir",         type=str, default=None,
                        help="특정 데이터셋 폴더 경로")
    parser.add_argument("--output_root", type=str, default=None,
                        help="H5 출력 루트 경로 (data/ 하위 전체 검증)")

    # 공통 옵션
    parser.add_argument("--dataset",     type=str, default=None,
                        help="검증할 데이터셋 (쉼표 구분, --output_root 모드에서 사용)")
    parser.add_argument("--sample",      type=int, default=None,
                        help="각 폴더에서 무작위 샘플링할 파일 수")
    parser.add_argument("--allow_nan_leads", action="store_true",
                        help="전체 NaN 리드를 오류로 취급하지 않음 (CODE-15% 등)")
    args = parser.parse_args()

    target_datasets = (
        [d.strip() for d in args.dataset.split(",")]
        if args.dataset else None
    )

    if args.file:
        # ── 단일 파일 상세 검증 ──────────────────────────────
        inspect_one(args.file)
        ds_name = Path(args.file).parent.name   # 폴더명으로 데이터셋 추정
        allow   = args.allow_nan_leads or ds_name in PARTIAL_LEAD_DATASETS
        issues  = validate_one(args.file, allow_nan_leads=allow)
        if issues:
            print(f"\n문제 {len(issues)}개:")
            for iss in issues:
                print(f"  ❌ {iss}")
        else:
            print(f"\n✅ 정상")

    elif args.dir:
        # ── 특정 폴더 일괄 검증 ─────────────────────────────
        ds_name = Path(args.dir).name
        allow   = args.allow_nan_leads or ds_name in PARTIAL_LEAD_DATASETS
        print(f"\n{'='*60}")
        print(f"  일괄 검증: {args.dir}")
        print(f"{'='*60}")
        result = batch_validate(
            args.dir,
            dataset_name   = ds_name,
            sample_n       = args.sample,
            allow_nan_leads= allow,
        )
        if result:
            print()
            print_batch_result(result, show_fail_n=20)

    elif args.output_root:
        # ── output_root 전체 검증 ────────────────────────────
        validate_output_root(
            args.output_root,
            target_datasets = target_datasets,
            sample_n        = args.sample,
            allow_nan_leads = args.allow_nan_leads,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()