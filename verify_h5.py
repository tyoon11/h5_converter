"""
HEEDB H5 검증 스크립트
======================
단일 파일 상세 검증 + 폴더 전체 일괄 검증

실행:
  python verify_h5.py --file data/he110745030.h5
  python verify_h5.py --dir /home/irteam/opendata1/h5/heedb/v4.0/data --sample 100
"""

import os
import sys
import argparse
import h5py
import numpy as np
from collections import Counter
from tqdm import tqdm

TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']

FIDUCIAL_FEATURE_KEYS = [
    "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
    "p_dur", "pr_seg", "qrs_dur", "st_seg", "t_dur",
    "pr_int", "qt_int", "rr_int", "tp_seg",
    "qtc_baz", "qtc_frid", "p_axis", "r_axis", "t_axis",
]


def inspect_one(h5_path):
    """단일 H5 파일 상세 출력"""
    print(f"\n{'='*60}")
    print(f"  {h5_path}")
    print(f"{'='*60}\n")

    with h5py.File(h5_path, "r") as f:
        # root attrs
        print("[root attrs]")
        for k in ["dataset_version", "file_name", "beat_ext_method", "fidu_extract_method"]:
            print(f"  {k}: {f.attrs.get(k, '(missing)')}")

        # metadata
        if "ECG" in f and "metadata" in f["ECG"]:
            meta = f["ECG/metadata"]
            print("\n[ECG/metadata attrs]")
            for k in ["record_name", "n_sig", "fs", "sig_len", "base_time", "base_date", "dtype"]:
                print(f"  {k}: {meta.attrs.get(k, '(missing)')}")
            print("[ECG/metadata datasets]")
            for k in ["sig_name", "fmt", "adc_gain", "baseline", "units", "adc_res", "adc_zero"]:
                if k in meta:
                    ds = meta[k]
                    print(f"  {k}: shape={ds.shape}, dtype={ds.dtype}, val={ds[()]}")

        # segments
        if "ECG" in f and "segments" in f["ECG"]:
            segs = f["ECG/segments"]
            seg_len = segs.attrs.get("seg_len", 0)
            print(f"\n[ECG/segments] seg_len={seg_len}")

            for i in range(seg_len):
                si = str(i)
                if si not in segs:
                    continue
                sg = segs[si]
                print(f"\n  --- segment {i} ---")

                # signal
                if "signal" in sg:
                    sig = sg["signal"]
                    print(f"  signal: shape={sig.shape}, dtype={sig.dtype}")
                    data = sig[()]
                    print(f"    min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}, "
                          f"mean={np.nanmean(data):.4f}, nan={np.sum(np.isnan(data))}")

                # beat_annotation
                if "beat_annotation" in sg:
                    ba = sg["beat_annotation"]
                    print(f"  beat_annotation:")
                    for k in ["sample", "symbol", "subtype", "chan", "num", "aux_note"]:
                        if k in ba:
                            print(f"    {k}: shape={ba[k].shape}, first5={ba[k][:5]}")

                # fiducial_point
                if "fiducial_point" in sg:
                    fp = sg["fiducial_point"]
                    print(f"  fiducial_point:")
                    for k in ["fsample", "fiducial"]:
                        if k in fp:
                            print(f"    {k}: shape={fp[k].shape}, first5={fp[k][:5]}")

                # fiducial_feature
                if "fiducial_feature" in sg:
                    ff = sg["fiducial_feature"]
                    print(f"  fiducial_feature:")
                    vals = {k: ff.attrs.get(k, "missing") for k in FIDUCIAL_FEATURE_KEYS}
                    nan_count = sum(1 for v in vals.values() if isinstance(v, (float, np.floating)) and np.isnan(v))
                    print(f"    {vals}")
                    print(f"    NaN count: {nan_count}/19")


def validate_one(h5_path):
    """단일 파일 필수 필드 검증, 문제 리스트 반환"""
    issues = []
    try:
        with h5py.File(h5_path, "r") as f:
            for k in ["dataset_version", "file_name"]:
                if k not in f.attrs:
                    issues.append(f"root: missing attr '{k}'")

            if "ECG" not in f:
                issues.append("missing ECG/")
                return issues

            if "metadata" not in f["ECG"]:
                issues.append("missing ECG/metadata/")
            else:
                meta = f["ECG/metadata"]
                for k in ["record_name", "n_sig", "fs", "sig_len"]:
                    if k not in meta.attrs:
                        issues.append(f"metadata: missing attr '{k}'")
                if "sig_name" in meta:
                    sn = list(meta["sig_name"][()])
                    sn_decoded = [s.decode() if isinstance(s, bytes) else s for s in sn]
                    if sn_decoded != TARGET_SIG_NAME:
                        issues.append(f"sig_name mismatch: {sn_decoded}")

            if "segments" not in f["ECG"]:
                issues.append("missing ECG/segments/")
            else:
                segs = f["ECG/segments"]
                seg_len = segs.attrs.get("seg_len", 0)
                for i in range(seg_len):
                    si = str(i)
                    if si not in segs:
                        issues.append(f"missing segment {i}")
                        continue
                    if "signal" not in segs[si]:
                        issues.append(f"segment {i}: missing signal")
                    else:
                        sig = segs[si]["signal"]
                        if sig.shape[0] != 12:
                            issues.append(f"segment {i}: signal shape[0]={sig.shape[0]} != 12")
    except Exception as e:
        issues.append(f"open error: {e}")

    return issues


def batch_validate(h5_dir, sample_n=None):
    """폴더 전체 일괄 검증"""
    files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]
    if sample_n and sample_n < len(files):
        import random
        random.seed(42)
        files = random.sample(files, sample_n)

    print(f"\n{'='*60}")
    print(f"  일괄 검증: {len(files)}개")
    print(f"{'='*60}\n")

    total = 0
    ok = 0
    fail_files = []
    shape_dist = Counter()
    fs_dist = Counter()

    for fname in tqdm(files, desc="검증"):
        path = os.path.join(h5_dir, fname)
        issues = validate_one(path)
        total += 1
        if issues:
            fail_files.append((fname, issues))
        else:
            ok += 1
            # 추가 통계
            try:
                with h5py.File(path, "r") as f:
                    fs_dist[int(f["ECG/metadata"].attrs["fs"])] += 1
                    if "0" in f["ECG/segments"]:
                        shape_dist[f["ECG/segments/0/signal"].shape] += 1
            except:
                pass

    print(f"\n결과: {ok}/{total} 정상, {len(fail_files)} 문제")

    if fail_files:
        print(f"\n문제 파일 (상위 10):")
        for fname, issues in fail_files[:10]:
            print(f"  {fname}: {issues}")

    if fs_dist:
        print(f"\nfs 분포: {dict(fs_dist)}")
    if shape_dist:
        print(f"signal shape 분포: {dict(shape_dist)}")


def main():
    parser = argparse.ArgumentParser(description="HEEDB H5 검증")
    parser.add_argument("--file", type=str, help="단일 H5 파일 상세 검증")
    parser.add_argument("--dir", type=str, help="폴더 일괄 검증")
    parser.add_argument("--sample", type=int, default=None, help="샘플 수 (dir 모드)")
    args = parser.parse_args()

    if args.file:
        inspect_one(args.file)
        issues = validate_one(args.file)
        if issues:
            print(f"\n문제 {len(issues)}개:")
            for i in issues:
                print(f"  - {i}")
        else:
            print(f"\n✅ 정상")
    elif args.dir:
        batch_validate(args.dir, args.sample)
    else:
        print("사용: --file <path> 또는 --dir <path>")


if __name__ == "__main__":
    main()