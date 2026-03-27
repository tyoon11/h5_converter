"""
Public ECG H5 검증 스크립트
=============================
실행:
  python verify_h5_public.py --file /data/h5/public/v1.0/data/ptb_xl/px12345_0.h5
  python verify_h5_public.py --dir /data/h5/public/v1.0/data/ptb_xl --sample 200
  python verify_h5_public.py --dir /data/h5/public/v1.0/data --recursive --sample 500
"""

import os
import sys
import argparse
import h5py
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']


# ═══════════════════════════════════════════════════════════════
# 단일 파일 검증
# ═══════════════════════════════════════════════════════════════
def validate_one(h5_path: str, allow_nan_leads: bool = False) -> list:
    """필수 필드 검증 → 문제 리스트 반환"""
    issues = []
    try:
        with h5py.File(h5_path, "r") as f:
            # root attrs
            for k in ["dataset_version", "file_name"]:
                if k not in f.attrs:
                    issues.append(f"root: missing attr '{k}'")

            if "ECG" not in f:
                issues.append("missing ECG/")
                return issues

            # metadata
            if "metadata" not in f["ECG"]:
                issues.append("missing ECG/metadata/")
            else:
                meta = f["ECG/metadata"]
                for k in ["record_name", "n_sig", "fs", "sig_len"]:
                    if k not in meta.attrs:
                        issues.append(f"metadata: missing attr '{k}'")
                if "sig_name" in meta:
                    sn = [s.decode() if isinstance(s, bytes) else s for s in meta["sig_name"][()]]
                    if sn != TARGET_SIG_NAME:
                        issues.append(f"sig_name 불일치: {sn}")

            # segments
            if "segments" not in f["ECG"]:
                issues.append("missing ECG/segments/")
                return issues

            segs = f["ECG/segments"]
            seg_len = segs.attrs.get("seg_len", 0)
            for i in range(seg_len):
                si = str(i)
                if si not in segs:
                    issues.append(f"segment {i} 누락")
                    continue
                sg = segs[si]
                if "signal" not in sg:
                    issues.append(f"segment {i}: signal 누락")
                else:
                    sig = sg["signal"][()]
                    if sig.shape[0] != 12:
                        issues.append(f"segment {i}: signal shape[0]={sig.shape[0]} ≠ 12")
                    nan_leads = int(np.sum(np.all(np.isnan(sig.astype(np.float32)), axis=1)))
                    if nan_leads > 0 and not allow_nan_leads:
                        issues.append(f"segment {i}: 전체 NaN 리드 {nan_leads}개")
                    elif nan_leads > 4:
                        issues.append(f"segment {i}: 과다 NaN 리드 {nan_leads}개")

    except Exception as e:
        issues.append(f"파일 열기 실패: {e}")
    return issues


def inspect_one(h5_path: str):
    """단일 파일 상세 출력"""
    print(f"\n{'='*60}")
    print(f"  {h5_path}")
    print(f"{'='*60}\n")
    with h5py.File(h5_path, "r") as f:
        print("[root attrs]")
        for k in ["dataset_version", "file_name", "beat_ext_method", "fidu_extract_method"]:
            print(f"  {k}: {f.attrs.get(k, '(missing)')}")

        if "ECG" in f and "metadata" in f["ECG"]:
            meta = f["ECG/metadata"]
            print("\n[ECG/metadata attrs]")
            for k in ["record_name", "n_sig", "fs", "sig_len", "base_date", "dtype"]:
                print(f"  {k}: {meta.attrs.get(k, '(missing)')}")
            if "sig_name" in meta:
                sn = [s.decode() if isinstance(s, bytes) else s for s in meta["sig_name"][()]]
                print(f"  sig_name: {sn}")
                print(f"  sig_name 일치: {sn == TARGET_SIG_NAME}")

        if "ECG" in f and "segments" in f["ECG"]:
            segs = f["ECG/segments"]
            seg_len = segs.attrs.get("seg_len", 0)
            print(f"\n[segments] seg_len={seg_len}")
            for i in range(seg_len):
                si = str(i)
                if si not in segs:
                    continue
                sg = segs[si]
                print(f"\n  --- segment {i} ---")
                if "signal" in sg:
                    sig = sg["signal"][()].astype(np.float32)
                    print(f"  signal: shape={sg['signal'].shape}, dtype={sg['signal'].dtype}")
                    nan_per_lead = np.sum(np.isnan(sig), axis=1)
                    full_nan = np.sum(np.all(np.isnan(sig), axis=1))
                    print(f"  전체 NaN 리드: {full_nan}/12")
                    print(f"  Lead별 NaN 샘플 수: {nan_per_lead.tolist()}")
                    valid = sig[~np.all(np.isnan(sig), axis=1)]
                    if len(valid):
                        print(f"  유효 리드 min={np.nanmin(valid):.4f}, max={np.nanmax(valid):.4f}, mean={np.nanmean(valid):.4f}")
                if "beat_annotation" in sg:
                    ba = sg["beat_annotation"]
                    print(f"  beat_annotation: {ba['sample'].shape[0]}개 R-peak")
                if "fiducial_feature" in sg:
                    ff = sg["fiducial_feature"]
                    nan_cnt = sum(1 for k in ff.attrs if np.isnan(float(ff.attrs[k])))
                    print(f"  fiducial_feature: NaN {nan_cnt}/19")


# ═══════════════════════════════════════════════════════════════
# 배치 검증
# ═══════════════════════════════════════════════════════════════
def batch_validate(h5_dir: str, sample_n: int = None, recursive: bool = False, allow_nan_leads: bool = False):
    """디렉토리 일괄 검증"""
    if recursive:
        files = []
        for root, _, fnames in os.walk(h5_dir):
            for fn in fnames:
                if fn.endswith(".h5"):
                    files.append(os.path.join(root, fn))
    else:
        files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith(".h5")]

    if sample_n and sample_n < len(files):
        import random
        random.seed(42)
        files = random.sample(files, sample_n)

    print(f"\n{'='*60}")
    print(f"  일괄 검증: {len(files):,}개 파일")
    print(f"  디렉토리: {h5_dir}")
    print(f"{'='*60}")

    ok = 0
    fail_files = []
    shape_dist = Counter()
    fs_dist = Counter()
    nan_lead_dist = Counter()
    dataset_stats = defaultdict(lambda: {"ok": 0, "fail": 0})

    for fpath in tqdm(files, desc="검증"):
        ds_name = os.path.basename(os.path.dirname(fpath))
        issues = validate_one(fpath, allow_nan_leads=allow_nan_leads)
        if issues:
            fail_files.append((fpath, issues))
            dataset_stats[ds_name]["fail"] += 1
        else:
            ok += 1
            dataset_stats[ds_name]["ok"] += 1
            try:
                with h5py.File(fpath, "r") as f:
                    fs_dist[int(f["ECG/metadata"].attrs.get("fs", 0))] += 1
                    if "0" in f["ECG/segments"]:
                        sig = f["ECG/segments/0/signal"][()].astype(np.float32)
                        shape_dist[sig.shape] += 1
                        full_nan = int(np.sum(np.all(np.isnan(sig), axis=1)))
                        nan_lead_dist[full_nan] += 1
            except Exception:
                pass

    total = len(files)
    print(f"\n결과: {ok:,}/{total:,} 정상 ({100*ok/total:.1f}%), {len(fail_files):,}개 문제")

    if dataset_stats:
        print("\n데이터셋별 결과:")
        for ds, stat in sorted(dataset_stats.items()):
            t = stat["ok"] + stat["fail"]
            print(f"  {ds:15s}: {stat['ok']:>6,}/{t:>6,} ({100*stat['ok']/max(t,1):.1f}%)")

    if fail_files:
        print(f"\n문제 파일 (상위 10개):")
        for fpath, issues in fail_files[:10]:
            print(f"  {os.path.basename(fpath)}: {issues}")

    if fs_dist:
        print(f"\nfs 분포: { {k: v for k, v in sorted(fs_dist.items())} }")
    if shape_dist:
        print(f"signal shape 분포 (상위 5):")
        for shape, cnt in Counter(shape_dist).most_common(5):
            print(f"  {shape}: {cnt:,}개")
    if nan_lead_dist:
        print(f"전체 NaN 리드 수 분포: {dict(sorted(nan_lead_dist.items()))}")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Public ECG H5 검증")
    parser.add_argument("--file", type=str, help="단일 H5 파일 상세 검증")
    parser.add_argument("--dir", type=str, help="디렉토리 일괄 검증")
    parser.add_argument("--sample", type=int, default=None, help="샘플 수 (dir 모드)")
    parser.add_argument("--recursive", action="store_true", help="하위 디렉토리 포함")
    parser.add_argument("--allow_nan_leads", action="store_true",
                        help="CODE-15% 등 일부 NaN 리드 허용")
    args = parser.parse_args()

    if args.file:
        inspect_one(args.file)
        issues = validate_one(args.file, allow_nan_leads=args.allow_nan_leads)
        if issues:
            print(f"\n문제 {len(issues)}개:")
            for i in issues:
                print(f"  - {i}")
        else:
            print(f"\n✅ 정상")
    elif args.dir:
        batch_validate(
            args.dir,
            sample_n=args.sample,
            recursive=args.recursive,
            allow_nan_leads=args.allow_nan_leads,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()