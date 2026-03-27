"""
공개 데이터셋 H5 변환 테스트
==============================
clinical_ts prepare_fn → npy 로드 → H5 저장 → 재로드 검증까지
단일 데이터셋의 첫 N개 레코드로 전체 파이프라인을 테스트합니다.

실행:
  python test_convert_public.py --dataset ptb_xl \\
      --dataset_dir /data/ecg_datasets \\
      --output_root /data/h5/public/v1.0 \\
      --n 3

  python test_convert_public.py --dataset code15 \\
      --dataset_dir /data/ecg_datasets \\
      --output_root /data/h5/public/v1.0 \\
      --n 1 --compute_beat --compute_fiducial
"""

import os
import sys
import argparse
import numpy as np
import h5py
from pathlib import Path

# 경로 설정
ROOT_DIR  = Path(__file__).resolve().parent
HEEDB_DIR = str(ROOT_DIR / "heedb")
CODE_DIR  = str(ROOT_DIR / "code")
sys.path.insert(0, HEEDB_DIR)
if os.path.isdir(CODE_DIR):
    sys.path.insert(0, CODE_DIR)

try:
    from clinical_ts.utils.ecg_utils import *               # noqa
    from clinical_ts.data.time_series_dataset_utils import *  # noqa
    CLINICAL_TS_OK = True
except ImportError:
    CLINICAL_TS_OK = False
    print("[ERROR] clinical_ts import 실패")
    sys.exit(1)

from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME
from utils_heedb import (
    extract_beat_annotation, extract_fiducial,
    signal_statistics, beat_similarity,
)
from convert_to_h5_public import (
    _build_configs, load_signal_from_npy,
    _default_age, _default_gender,
    CANONICAL_TO_TARGET_IDX, CODE15_TARGET_POS,
)

FIDUCIAL_FEATURE_KEYS = [
    "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
    "p_dur", "pr_seg", "qrs_dur", "st_seg", "t_dur",
    "pr_int", "qt_int", "rr_int", "tp_seg",
    "qtc_baz", "qtc_frid", "p_axis", "r_axis", "t_axis",
]

# CODE-15%처럼 일부 리드가 NaN인 데이터셋
PARTIAL_LEAD_DATASETS = {"code15"}


# ═══════════════════════════════════════════════════════════════
# 단일 데이터셋 테스트
# ═══════════════════════════════════════════════════════════════
def test_dataset(args):
    dataset_name = args.dataset
    dataset_dir  = Path(args.dataset_dir)
    output_root  = Path(args.output_root)
    n            = args.n

    configs = _build_configs(dataset_dir, output_root)
    if dataset_name not in configs:
        print(f"[ERROR] 알 수 없는 데이터셋: '{dataset_name}'")
        print(f"  선택 가능: {list(configs.keys())}")
        sys.exit(1)

    cfg        = configs[dataset_name]
    prefix     = cfg["prefix"]
    fs         = cfg["fs"]
    n_channels = cfg["n_channels"]
    target_dir = Path(cfg["target_dir"])
    data_dir   = Path(cfg["data_dir"])
    h5_dir     = output_root / "data" / dataset_name
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(h5_dir, exist_ok=True)

    allow_nan_leads = dataset_name in PARTIAL_LEAD_DATASETS

    print(f"\n{'='*60}")
    print(f"  테스트: {dataset_name.upper()}  (fs={fs} Hz, prefix={prefix})")
    print(f"{'='*60}")

    # ── Step 1: prepare_fn 실행 ──────────────────────────────
    prepare_fn_name = cfg["prepare_fn"]
    prepare_fn      = globals().get(prepare_fn_name)
    if prepare_fn is None:
        print(f"[ERROR] 함수 '{prepare_fn_name}' 없음")
        sys.exit(1)

    print(f"\n[1] {prepare_fn_name}() 실행 중...")
    kwargs = {"data_path": data_dir, "target_folder": target_dir}
    kwargs.update(cfg.get("prepare_kwargs", {}))
    try:
        df, lbl_map, lbl_itos, _ = prepare_fn(**kwargs)
    except Exception as e:
        print(f"[ERROR] prepare 실패: {e}")
        sys.exit(1)

    print(f"  df shape   : {df.shape}")
    print(f"  columns    : {list(df.columns)}")
    print(f"  첫 행 샘플 : {df.iloc[0].to_dict()}")
    if lbl_itos:
        print(f"  레이블 수  : {len(lbl_itos)}")

    # ── Step 2: 첫 n개 레코드 처리 ──────────────────────────
    print(f"\n[2] 첫 {n}개 레코드 변환 테스트")

    saved_paths   = []
    saved_sigs    = []

    for i, (_, row) in enumerate(df.head(n).iterrows()):
        row_dict = row.to_dict()
        npy_path = str(row_dict.get("data", ""))
        pid      = str(row_dict.get("patient_id", row_dict.get("pid", i))).strip()
        age      = _default_age(row_dict)
        gender   = _default_gender(row_dict)

        print(f"\n  ── 레코드 {i} ──")
        print(f"  pid     : {pid}")
        print(f"  age     : {age:.4f}  gender: {gender}")
        print(f"  npy     : {npy_path}")

        if not npy_path or not os.path.isfile(npy_path):
            print(f"  [SKIP] npy 파일 없음")
            continue

        # 신호 로드 & 채널 reorder
        try:
            sig = load_signal_from_npy(npy_path, n_channels, dataset_name)
        except Exception as e:
            print(f"  [ERROR] 신호 로드 실패: {e}")
            continue

        sig_f32     = sig.astype(np.float32)
        nan_leads   = int(np.sum(np.all(np.isnan(sig_f32), axis=1)))
        valid_leads = 12 - nan_leads

        print(f"  signal shape : {sig.shape}  dtype: {sig.dtype}")
        print(f"  유효 리드    : {valid_leads}/12  (전체 NaN 리드: {nan_leads})")
        if valid_leads > 0:
            print(f"  값 범위      : min={np.nanmin(sig_f32):.4f}  max={np.nanmax(sig_f32):.4f}")

        sig_valid = np.nan_to_num(sig_f32)

        # beat annotation
        ba, beat_method = None, ""
        if args.compute_beat:
            print(f"  beat_annotation 추출 중...")
            try:
                ba          = extract_beat_annotation(sig_valid[1], fs)
                beat_method = "neurokit2"
                print(f"    R-peaks: {len(ba['sample'])}개")
            except Exception as e:
                print(f"    [WARNING] beat 추출 실패: {e}")

        # fiducial
        fp, ff, fidu_method = None, None, ""
        if args.compute_fiducial:
            print(f"  fiducial 추출 중...")
            try:
                fp, ff      = extract_fiducial(sig_valid, fs)
                fidu_method = "neurokit2-dwt"
                nan_cnt = sum(1 for v in ff.values()
                              if isinstance(v, (float, np.floating)) and np.isnan(v))
                print(f"    fiducial points : {len(fp['fsample'])}개")
                print(f"    feature NaN     : {nan_cnt}/19")
            except Exception as e:
                print(f"    [WARNING] fiducial 추출 실패: {e}")

        # signal statistics
        print(f"  signal statistics 계산 중...")
        try:
            stats = signal_statistics(sig_valid.T)
            amp_std_valid = [
                f"{v:.4f}" if not np.isnan(v) else "NaN"
                for v in stats["amp_std"]
            ]
            print(f"    amp_std: {amp_std_valid}")
        except Exception as e:
            print(f"    [WARNING] statistics 실패: {e}")

        # H5 저장
        file_name = f"{prefix}{pid}_{i}"
        h5_path   = str(h5_dir / f"{file_name}.h5")

        print(f"  H5 저장: {h5_path}")
        try:
            with h5py.File(h5_path, "w") as h5f:
                create_h5_structure(
                    h5f,
                    file_name        = file_name,
                    beat_ext_method  = beat_method,
                    fidu_extract_method = fidu_method,
                    record_name      = str(row_dict.get("record_name", file_name)),
                    n_sig=12, fs=fs, sig_len=sig.shape[1],
                    base_date        = str(row_dict.get("base_date",
                                          row_dict.get("acquisition_date", ""))),
                    sig_name         = TARGET_SIG_NAME,
                    signal           = [sig],
                    seg_len          = 1,
                    beat_annotation  = [ba] if ba else None,
                    fiducial_point   = [fp] if fp else None,
                    fiducial_feature = [ff] if ff else None,
                )
            saved_paths.append(h5_path)
            saved_sigs.append(sig)
            print(f"  ✅ 저장 완료")
        except Exception as e:
            print(f"  [ERROR] H5 저장 실패: {e}")

    # ── Step 3: 저장된 H5 재로드 검증 ──────────────────────
    if not saved_paths:
        print("\n[ERROR] 저장된 파일이 없습니다.")
        return

    print(f"\n[3] H5 재로드 검증")

    all_pass = True
    for h5_path, orig_sig in zip(saved_paths, saved_sigs):
        print(f"\n  파일: {Path(h5_path).name}")
        errors = []

        try:
            with h5py.File(h5_path, "r") as f:
                # root attrs
                for k in ["dataset_version", "file_name"]:
                    if k not in f.attrs:
                        errors.append(f"root attr '{k}' 누락")
                print(f"  file_name : {f.attrs.get('file_name', '?')}")

                # metadata
                meta = f["ECG/metadata"]
                fs_h5     = int(meta.attrs.get("fs", 0))
                sig_len   = int(meta.attrs.get("sig_len", 0))
                sn        = [s.decode() if isinstance(s, bytes) else s
                             for s in meta["sig_name"][()]]
                sn_ok     = sn == TARGET_SIG_NAME
                if not sn_ok:
                    errors.append(f"sig_name 불일치: {sn}")
                print(f"  fs={fs_h5}  sig_len={sig_len}  sig_name 일치={sn_ok}")

                # signal
                s0        = f["ECG/segments/0"]
                sig_h5    = s0["signal"][()]
                shape_ok  = sig_h5.shape[0] == 12
                if not shape_ok:
                    errors.append(f"signal shape[0]={sig_h5.shape[0]} ≠ 12")

                # 원본과 값 일치 확인 (NaN 무시)
                orig_f32  = orig_sig.astype(np.float32)
                h5_f32    = sig_h5.astype(np.float32)
                mask      = ~(np.isnan(orig_f32) | np.isnan(h5_f32))
                if mask.any():
                    val_ok = np.allclose(orig_f32[mask], h5_f32[mask], atol=0.01)
                else:
                    val_ok = True   # 전부 NaN이면 패스
                if not val_ok:
                    errors.append("signal 값 불일치 (atol=0.01)")

                nan_leads = int(np.sum(np.all(np.isnan(h5_f32), axis=1)))
                print(f"  signal shape={sig_h5.shape}  dtype={sig_h5.dtype}"
                      f"  전체NaN리드={nan_leads}  값일치={val_ok}")

                # beat_annotation
                if "beat_annotation" in s0:
                    n_peaks = s0["beat_annotation/sample"].shape[0]
                    print(f"  beat_annotation: R-peak {n_peaks}개")

                # fiducial_feature
                if "fiducial_feature" in s0:
                    ff_h5   = s0["fiducial_feature"]
                    nan_cnt = sum(1 for k in FIDUCIAL_FEATURE_KEYS
                                  if np.isnan(float(ff_h5.attrs.get(k, float("nan")))))
                    print(f"  fiducial_feature: NaN {nan_cnt}/19")

        except Exception as e:
            errors.append(f"파일 읽기 실패: {e}")

        if errors:
            all_pass = False
            for err in errors:
                print(f"  ❌ {err}")
        else:
            if not allow_nan_leads and nan_leads > 0:
                print(f"  ⚠️  전체 NaN 리드 {nan_leads}개 (CODE-15% 등 일부 데이터셋은 정상)")
            else:
                print(f"  ✅ 정상")

    print(f"\n{'='*60}")
    if all_pass:
        print(f"  ✅ 전체 통과 ({len(saved_paths)}/{n}개)")
    else:
        print(f"  ❌ 일부 실패 — 위 오류 메시지 확인")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="공개 데이터셋 H5 변환 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python test_convert_public.py --dataset ptb_xl \\
      --dataset_dir /data/ecg_datasets --output_root /data/h5/public/v1.0

  python test_convert_public.py --dataset code15 \\
      --dataset_dir /data/ecg_datasets --output_root /data/h5/public/v1.0 \\
      --n 2 --compute_beat --compute_fiducial
        """,
    )
    parser.add_argument("--dataset",      type=str, required=True,
                        help="테스트할 데이터셋 키 (예: ptb_xl, mimic, code15)")
    parser.add_argument("--dataset_dir",  type=str, required=True,
                        help="원본 데이터셋 루트 경로")
    parser.add_argument("--output_root",  type=str, required=True,
                        help="H5 출력 루트 경로")
    parser.add_argument("--n",            type=int, default=3,
                        help="테스트할 레코드 수 (기본 3)")
    parser.add_argument("--compute_beat",     action="store_true",
                        help="beat_annotation 추출 포함")
    parser.add_argument("--compute_fiducial", action="store_true",
                        help="fiducial_point/feature 추출 포함")
    args = parser.parse_args()

    test_dataset(args)


if __name__ == "__main__":
    main()