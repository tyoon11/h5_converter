"""
Public ECG H5 변환 테스트
===========================
clinical_ts prepare_fn 결과 → H5 저장 → 검증까지 단일 데이터셋으로 테스트.

실행:
  python test_convert_public.py --dataset ptb_xl \
      --dataset_dir /data/ecg_datasets \
      --output_h5 /tmp/test_public.h5 \
      --n 3
"""

import os
import sys
import argparse
import numpy as np
import h5py
from pathlib import Path

CODE_DIR = str(Path(__file__).resolve().parent / "code")
if os.path.isdir(CODE_DIR):
    sys.path.insert(0, CODE_DIR)

try:
    from clinical_ts.utils.ecg_utils import *          # noqa
    from clinical_ts.data.time_series_dataset_utils import *  # noqa
    CLINICAL_TS_OK = True
except ImportError:
    CLINICAL_TS_OK = False
    print("[ERROR] clinical_ts import 실패")
    sys.exit(1)

from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME
from convert_to_h5_public import _build_configs, load_signal_from_npy, _default_age, _default_gender
from utils_heedb import extract_beat_annotation, extract_fiducial, signal_statistics


# ───────────────────────────────────────────────────────────────
def test_dataset(dataset_name: str, dataset_dir: Path, output_h5: str, n: int = 3):
    configs = _build_configs(dataset_dir, Path("/tmp/public_test_target"))
    if dataset_name not in configs:
        print(f"[ERROR] 알 수 없는 데이터셋: {dataset_name}")
        print(f"  선택 가능: {list(configs.keys())}")
        return

    cfg = configs[dataset_name]
    prefix = cfg["prefix"]
    fs = cfg["fs"]
    n_channels = cfg["n_channels"]
    target_dir = Path(cfg["target_dir"])
    os.makedirs(target_dir, exist_ok=True)

    # ── Step 1: prepare_fn 실행 ─────────────────────────────
    prepare_fn_name = cfg["prepare_fn"]
    prepare_fn = globals().get(prepare_fn_name)
    if prepare_fn is None:
        print(f"[ERROR] 함수 '{prepare_fn_name}' 없음")
        return

    print(f"\n[1] {prepare_fn_name}() 실행...")
    kwargs = {"data_path": Path(cfg["data_dir"]), "target_folder": target_dir}
    kwargs.update(cfg.get("prepare_kwargs", {}))
    df, lbl_map, lbl_itos, _ = prepare_fn(**kwargs)
    print(f"  df shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  첫 행: {df.iloc[0].to_dict()}")

    # ── Step 2: 첫 n개 레코드 변환 + 저장 ───────────────────
    print(f"\n[2] 첫 {n}개 레코드 H5 변환...")
    saved_paths = []

    for i, (_, row) in enumerate(df.head(n).iterrows()):
        row_dict = row.to_dict()
        npy_path = row_dict.get("data", "")
        print(f"\n  --- 레코드 {i} ---")
        print(f"  npy_path: {npy_path}")
        print(f"  age: {row_dict.get('age', 'N/A')}, sex: {row_dict.get('sex', row_dict.get('gender', 'N/A'))}")

        if not npy_path or not os.path.isfile(str(npy_path)):
            print(f"  [SKIP] npy 파일 없음")
            continue

        try:
            sig = load_signal_from_npy(str(npy_path), n_channels, dataset_name)
        except Exception as e:
            print(f"  [ERROR] 신호 로드 실패: {e}")
            continue

        print(f"  signal shape: {sig.shape}, dtype: {sig.dtype}")
        sig_f32 = sig.astype(np.float32)
        nan_leads = np.sum(np.all(np.isnan(sig_f32), axis=1))
        print(f"  전체 NaN 리드: {nan_leads}/12")
        valid_sig = np.nan_to_num(sig_f32)
        print(f"  min={np.nanmin(sig_f32):.4f}, max={np.nanmax(sig_f32):.4f}")

        # beat annotation
        print(f"  beat_annotation 추출 중...")
        ba = extract_beat_annotation(valid_sig[1], fs)   # Lead II
        print(f"    R-peaks: {len(ba['sample'])}개")

        # fiducial
        print(f"  fiducial 추출 중...")
        fp, ff = extract_fiducial(valid_sig, fs)
        print(f"    fiducial points: {len(fp['fsample'])}개")
        print(f"    fiducial features NaN: {sum(1 for v in ff.values() if isinstance(v, float) and np.isnan(v))}/19")

        # signal statistics
        stats = signal_statistics(valid_sig.T)
        print(f"  amp_std: {[f'{v:.4f}' for v in stats['amp_std']]}")

        # H5 저장
        pid = str(row_dict.get("patient_id", row_dict.get("pid", i))).strip()
        file_name = f"{prefix}{pid}_{i}"
        h5_path = str(Path(output_h5).parent / f"{file_name}.h5") if n > 1 else output_h5
        saved_paths.append(h5_path)

        age = _default_age(row_dict)
        gender = _default_gender(row_dict)

        with h5py.File(h5_path, "w") as h5f:
            create_h5_structure(
                h5f,
                file_name=file_name,
                beat_ext_method="neurokit2",
                fidu_extract_method="neurokit2-dwt",
                record_name=str(row_dict.get("record_name", file_name)),
                n_sig=12, fs=fs, sig_len=sig.shape[1],
                base_date=str(row_dict.get("base_date", row_dict.get("acquisition_date", ""))),
                sig_name=TARGET_SIG_NAME,
                signal=[sig],
                seg_len=1,
                beat_annotation=[ba],
                fiducial_point=[fp],
                fiducial_feature=[ff],
            )
        print(f"  ✅ 저장: {h5_path}")

    # ── Step 3: 검증 ────────────────────────────────────────
    print(f"\n[3] 저장된 H5 검증...")
    for h5_path in saved_paths:
        if not os.path.exists(h5_path):
            continue
        print(f"\n  파일: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            print(f"  file_name: {f.attrs.get('file_name', '?')}")
            meta = f["ECG/metadata"]
            sn = [s.decode() if isinstance(s, bytes) else s for s in meta["sig_name"][()]]
            print(f"  sig_name 일치: {sn == TARGET_SIG_NAME}")
            print(f"  fs={meta.attrs['fs']}, sig_len={meta.attrs['sig_len']}")
            seg0 = f["ECG/segments/0"]
            sig_h5 = seg0["signal"][()]
            print(f"  signal shape: {sig_h5.shape}, dtype: {sig_h5.dtype}")
            if "beat_annotation" in seg0:
                print(f"  R-peaks: {seg0['beat_annotation/sample'].shape[0]}개")
            if "fiducial_feature" in seg0:
                nan_cnt = sum(1 for k in seg0["fiducial_feature"].attrs
                              if np.isnan(float(seg0["fiducial_feature"].attrs[k])))
                print(f"  fiducial features NaN: {nan_cnt}/19")

    print(f"\n✅ 테스트 완료 ({dataset_name})")


def main():
    parser = argparse.ArgumentParser(description="Public ECG H5 변환 테스트")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_h5", type=str, default="/tmp/test_public.h5")
    parser.add_argument("--n", type=int, default=3, help="테스트할 레코드 수")
    args = parser.parse_args()

    test_dataset(args.dataset, Path(args.dataset_dir), args.output_h5, args.n)


if __name__ == "__main__":
    main()