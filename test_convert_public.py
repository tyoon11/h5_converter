"""
공개 데이터셋 H5 변환 테스트 (데이터셋별 1개)
=============================================
각 데이터셋의 첫 번째 유효 레코드를 변환하고
H5 재로드 검증까지 전체 파이프라인을 확인합니다.

실행:
  # physionet 8개 테스트
  python test_convert_public.py --group physionet \\
      --physionet_root /home/irteam/ddn-opendata1/raw/physionet.org/files \\
      --output_root    /tmp/h5_public_test

  # ZZU 테스트
  python test_convert_public.py --group zzu \\
      --zzu_root    /home/irteam/ddn-opendata1/raw/ZZU-pECG \\
      --output_root /tmp/h5_public_test

  # 특정 데이터셋만
  python test_convert_public.py --dataset georgia,ptbxl \\
      --physionet_root ... --output_root ...

  # beat/fiducial 포함
  python test_convert_public.py --dataset georgia \\
      --physionet_root ... --output_root ... \\
      --compute_beat --compute_fiducial
"""

import os
import sys
import argparse
import numpy as np
import h5py
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent
HEEDB_DIR = str(ROOT_DIR / "heedb")
sys.path.insert(0, HEEDB_DIR)

import wfdb
from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME
from utils_heedb import extract_beat_annotation, extract_fiducial, signal_statistics
from convert_to_h5_public import (
    _build_configs, _make_table_df, _make_filename_df,
    LEAD_ALIASES, TARGET_SET,
    PHYSIONET_DATASETS, ZZU_DATASETS,
)

FIDUCIAL_FEATURE_KEYS = [
    "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
    "p_dur", "pr_seg", "qrs_dur", "st_seg", "t_dur",
    "pr_int", "qt_int", "rr_int", "tp_seg",
    "qtc_baz", "qtc_frid", "p_axis", "r_axis", "t_axis",
]
P = "✅"
F = "❌"
W = "⚠️ "


# ═══════════════════════════════════════════════════════════════
# 첫 번째 유효 레코드 탐색
# ═══════════════════════════════════════════════════════════════
def _find_valid(records: list, max_try: int = 10) -> tuple:
    """유효한 (rec, sig, d, raw_names, rec_info) 반환. 없으면 None."""
    for rec_info in records[:max_try]:
        try:
            rec = wfdb.rdrecord(rec_info["record_path"])
        except Exception:
            continue
        sig = rec.p_signal
        if sig is None:
            sig = rec.d_signal
        if sig is None:
            continue
        d         = rec.__dict__
        raw_names = [LEAD_ALIASES.get(n, n) for n in rec.sig_name]
        if (d["n_sig"] != 12
                or set(raw_names) != TARGET_SET
                or sig.shape[0] / d["fs"] < 1.0
                or np.any(np.all(sig == 0, axis=0))):
            continue
        return rec, sig, d, raw_names, rec_info
    return None


# ═══════════════════════════════════════════════════════════════
# 단일 데이터셋 테스트
# ═══════════════════════════════════════════════════════════════
def test_one(dataset_name: str, cfg: dict, output_root: Path, args) -> dict:
    prefix  = cfg["prefix"]
    # flat: 모든 데이터셋을 data/ 아래 같은 폴더에 저장
    h5_dir  = output_root / "data"
    os.makedirs(h5_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{cfg['name']}]  prefix={prefix}")
    print(f"{'='*60}")

    # ── 1. 레코드 목록 ───────────────────────────────────────
    print("[1] 레코드 목록")
    try:
        records = cfg["records_fn"]()
    except Exception as e:
        print(f"  {F} 실패: {e}")
        return {"dataset": dataset_name, "status": "FAIL", "reason": str(e)}
    print(f"  전체: {len(records):,}개")

    # ── 2. 유효 레코드 탐색 ──────────────────────────────────
    print("[2] 유효 레코드 탐색 (최대 10개 시도)")
    found = _find_valid(records)
    if found is None:
        print(f"  {F} 유효 레코드 없음")
        return {"dataset": dataset_name, "status": "FAIL", "reason": "no valid record"}
    rec, sig, d, raw_names, rec_info = found

    fs      = d["fs"]
    sig_len = d["sig_len"]
    print(f"  record_path  : {rec_info['record_path']}")
    print(f"  pid / rid    : {rec_info['pid']} / {rec_info['rid']}")
    print(f"  fs           : {fs} Hz")
    print(f"  shape        : {sig.shape}  ({sig.shape[0]/fs:.1f}s)")
    print(f"  leads (norm) : {raw_names}")
    print(f"  age / gender : {rec_info['age']} / {rec_info['gender']}")

    # ── 3. Reorder ───────────────────────────────────────────
    print("[3] Reorder → (12, samples) fp16")
    idx           = [raw_names.index(n) for n in TARGET_SIG_NAME]
    sig_reordered = sig[:, idx].T.astype(np.float16)
    sig_f32       = sig_reordered.astype(np.float32)
    nan_leads     = int(np.sum(np.all(np.isnan(sig_f32), axis=1)))
    print(f"  shape        : {sig_reordered.shape}  dtype={sig_reordered.dtype}")
    print(f"  값 범위      : min={np.nanmin(sig_f32):.4f}  max={np.nanmax(sig_f32):.4f}")
    print(f"  전체NaN 리드 : {nan_leads}/12")

    # ── 4. signal_statistics ─────────────────────────────────
    print("[4] signal_statistics")
    try:
        stats = signal_statistics(sig_f32.T)
        print(f"  amp_std  : {[f'{v:.3f}' for v in stats['amp_std']]}")
    except Exception as e:
        print(f"  {W} {e}")

    # ── 5. beat_annotation (옵션) ────────────────────────────
    ba_list, beat_method = None, ""
    if args.compute_beat:
        print("[5] beat_annotation")
        try:
            ba          = extract_beat_annotation(np.nan_to_num(sig_f32[1]), fs)
            ba_list     = [ba]
            beat_method = "neurokit2"
            print(f"  R-peaks  : {len(ba['sample'])}개  {P}")
        except Exception as e:
            print(f"  {W} {e}")

    # ── 6. fiducial (옵션) ───────────────────────────────────
    fp_list, ff_list, fidu_method = None, None, ""
    if args.compute_fiducial:
        print("[6] fiducial")
        try:
            fp, ff      = extract_fiducial(np.nan_to_num(sig_f32), fs)
            fp_list     = [fp]
            ff_list     = [ff]
            fidu_method = "neurokit2-dwt"
            nan_cnt     = sum(1 for v in ff.values()
                              if isinstance(v, (float, np.floating)) and np.isnan(v))
            print(f"  points   : {len(fp['fsample'])}개")
            print(f"  feat NaN : {nan_cnt}/19  {P}")
        except Exception as e:
            print(f"  {W} {e}")

    # ── 7. H5 저장 ───────────────────────────────────────────
    age    = rec_info.get("age",    -1.0)
    gender = rec_info.get("gender",  0)
    # WFDB 헤더 보완
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
    h5_path   = h5_dir / f"{file_name}.h5"

    print(f"[7] H5 저장: {h5_path.name}")
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
        print(f"  {P} 저장 완료")
    except Exception as e:
        print(f"  {F} 저장 실패: {e}")
        return {"dataset": dataset_name, "status": "FAIL", "reason": str(e)}

    # ── 8. H5 검증 ───────────────────────────────────────────
    print("[8] H5 검증")
    errors = []
    sig_h5_shape = None
    try:
        with h5py.File(h5_path, "r") as f:
            fn_h5 = f.attrs.get("file_name", "")
            fn_ok = fn_h5 == file_name
            print(f"  file_name     : {fn_h5}  {P if fn_ok else F}")
            if not fn_ok:
                errors.append(f"file_name 불일치: {fn_h5}")

            meta  = f["ECG/metadata"]
            fs_h5 = int(meta.attrs.get("fs", 0))
            sn    = [s.decode() if isinstance(s, bytes) else s for s in meta["sig_name"][()]]
            sn_ok = sn == TARGET_SIG_NAME
            print(f"  fs            : {fs_h5}  {P if fs_h5==fs else F}")
            print(f"  sig_name      : {P if sn_ok else F}  {sn if not sn_ok else ''}")
            if not sn_ok:
                errors.append(f"sig_name 불일치")

            s0          = f["ECG/segments/0"]
            sig_h5      = s0["signal"][()]
            sig_h5_shape = sig_h5.shape
            sh_ok       = sig_h5.shape[0] == 12
            print(f"  signal shape  : {sig_h5.shape}  dtype={sig_h5.dtype}  {P if sh_ok else F}")
            if not sh_ok:
                errors.append(f"shape[0]={sig_h5.shape[0]} ≠ 12")

            orig = sig_reordered.astype(np.float32)
            h5v  = sig_h5.astype(np.float32)
            mask = ~(np.isnan(orig) | np.isnan(h5v))
            val_ok = np.allclose(orig[mask], h5v[mask], atol=0.01) if mask.any() else True
            print(f"  값 일치       : {P if val_ok else F}")
            if not val_ok:
                errors.append("signal 값 불일치")

            if "beat_annotation" in s0:
                print(f"  beat_annotation: {s0['beat_annotation/sample'].shape[0]}개  {P}")
            if "fiducial_feature" in s0:
                nc = sum(1 for k in FIDUCIAL_FEATURE_KEYS
                         if np.isnan(float(s0["fiducial_feature"].attrs.get(k, float("nan")))))
                print(f"  fiducial_feat  : NaN {nc}/19  {P}")

    except Exception as e:
        errors.append(f"재로드 실패: {e}")

    status = "PASS" if not errors else "FAIL"
    icon   = P if status == "PASS" else F
    print(f"\n  {icon} {dataset_name}: {status}")
    for err in errors:
        print(f"    {F} {err}")

    return {
        "dataset":   dataset_name,
        "status":    status,
        "file_name": file_name,
        "fs":        fs,
        "shape":     sig_h5_shape,
        "age":       age,
        "gender":    gender,
        "errors":    errors,
    }


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="공개 데이터셋 H5 변환 테스트 (데이터셋별 1개)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target_grp = parser.add_mutually_exclusive_group()
    target_grp.add_argument("--group",   type=str, choices=["physionet", "zzu", "all"])
    target_grp.add_argument("--dataset", type=str, help="쉼표 구분: georgia,ptbxl")

    parser.add_argument("--physionet_root", type=str,
                        default="/home/irteam/ddn-opendata1/raw/physionet.org/files")
    parser.add_argument("--zzu_root",       type=str,
                        default="/home/irteam/ddn-opendata1/raw/ZZU-pECG")
    parser.add_argument("--output_root",    type=str, default="/tmp/h5_public_test")
    parser.add_argument("--compute_beat",     action="store_true")
    parser.add_argument("--compute_fiducial", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    configs = _build_configs(args.physionet_root, args.zzu_root)

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
            print(f"알 수 없는 데이터셋: {unknown}\n선택 가능: {list(configs.keys())}")
            sys.exit(1)
    else:
        parser.print_help()
        return

    results = []
    for ds_name in target_datasets:
        result = test_one(ds_name, configs[ds_name], output_root, args)
        results.append(result)

    # 최종 요약
    print(f"\n\n{'='*70}")
    print(f"  최종 요약")
    print(f"{'='*70}")
    print(f"  {'데이터셋':<20} {'상태':<6} {'file_name':<28} {'fs':>5}  {'shape':<16} age / gender")
    print(f"  {'-'*70}")
    for r in results:
        icon  = P if r["status"] == "PASS" else (W if r["status"] == "SKIP" else F)
        print(f"  {icon} {r['dataset']:<18} {r['status']:<6} "
              f"{r.get('file_name','-'):<28} {str(r.get('fs','-')):>5}  "
              f"{str(r.get('shape','-')):<16} "
              f"{r.get('age',-1):.4f} / {r.get('gender',0)}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] != "PASS")
    print(f"\n  {P} PASS {n_pass}  {F} FAIL/SKIP {n_fail}  / 전체 {len(results)}")
    print(f"  출력: {output_root}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()