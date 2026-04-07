"""
통합 변환 테스트 (데이터셋별 1개 레코드)
==========================================
모든 데이터셋(HEEDB + PhysioNet + ZZU)에 대해
첫 번째 유효 레코드를 변환하고 H5 재로드 검증까지 전체 파이프라인을 확인합니다.

실행:
  # 공개 데이터셋 전체
  python test_convert.py --group physionet \\
      --physionet_root /data/raw/physionet.org/files \\
      --output_root /tmp/h5_test

  # ZZU
  python test_convert.py --group zzu \\
      --zzu_root /data/raw/ZZU-pECG \\
      --output_root /tmp/h5_test

  # HEEDB
  python test_convert.py --group heedb \\
      --heedb_root /data/raw/heedb/ECG \\
      --output_root /tmp/h5_test

  # 전체
  python test_convert.py --group all \\
      --heedb_root /data/raw/heedb/ECG \\
      --physionet_root /data/raw/physionet.org/files \\
      --zzu_root /data/raw/ZZU-pECG \\
      --output_root /tmp/h5_test

  # 특정 데이터셋만
  python test_convert.py --dataset georgia,ptbxl,heedb_i0001 \\
      --physionet_root ... --heedb_root ... --output_root ...

  # beat/fiducial 포함
  python test_convert.py --dataset ptbxl \\
      --physionet_root ... --output_root ... \\
      --compute_beat --compute_fiducial
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, SCRIPT_DIR)

import wfdb
from utils.h5_structure import create_h5_structure, TARGET_SIG_NAME, FIDUCIAL_FEATURE_KEYS
from utils.signal_processing import (
    extract_beat_annotation, extract_fiducial, signal_statistics, has_zero_lead,
)
from convert_to_h5 import (
    _build_all_configs, _make_table_df,
    PHYSIONET_DATASETS, ZZU_DATASETS, HEEDB_DATASETS,
    LEAD_ALIASES, TABLE_COLS,
)

# HEEDB metadata.csv는 수백만 행이므로 테스트 시 앞부분만 읽음
HEEDB_TEST_NROWS = 200


def _patch_heedb_configs(configs: dict, heedb_root: str):
    """HEEDB records_fn을 nrows=HEEDB_TEST_NROWS로 제한된 버전으로 교체."""
    import os
    import pandas as pd
    from convert_to_h5 import encode_gender, normalize_pid

    inst_map = {
        "heedb_i0001": ("I0001", "SexDSC"),
        "heedb_i0006": ("I0006", "Sex"),
    }

    for ds_name, (inst_code, gender_field) in inst_map.items():
        if ds_name not in configs:
            continue
        base_dir  = os.path.join(heedb_root, inst_code)
        wfdb_root = os.path.join(base_dir, "WFDB")
        meta_path = os.path.join(base_dir, "metadata", "metadata.csv")

        def _make_fn(_wfdb=wfdb_root, _meta=meta_path, _gf=gender_field):
            def _fn():
                if not os.path.exists(_meta):
                    return []
                df = pd.read_csv(_meta, dtype=str, low_memory=False,
                                 nrows=HEEDB_TEST_NROWS)
                records = []
                for rid, row in df.iterrows():
                    pid    = normalize_pid(row.get("BDSPPatientID", ""))
                    fn_raw = str(row.get("FileName", "")).strip().lstrip("/")
                    fn_clean = fn_raw[5:] if fn_raw.startswith("WFDB/") else fn_raw
                    if not pid or pid == "nan":
                        continue
                    try:
                        age = round(float(row.get("AgeAtAcquisition", -1)) / 365.25 / 100.0, 6)
                    except (TypeError, ValueError):
                        age = -1.0
                    records.append({
                        "record_path": os.path.join(_wfdb, fn_clean),
                        "pid": pid, "rid": rid,
                        "age": age,
                        "gender": encode_gender(str(row.get(_gf, ""))),
                        "source": "heedb",
                    })
                return records
            return _fn

        configs[ds_name]["records_fn"] = _make_fn()

TARGET_SET = set(TARGET_SIG_NAME)
PASS = "OK"
FAIL = "FAIL"
WARN = "WARN"


# ═══════════════════════════════════════════════════════════════
# 유효 레코드 탐색
# ═══════════════════════════════════════════════════════════════
def _find_valid(records: list, max_try: int = 20):
    """유효한 레코드를 탐색합니다. 없으면 None 반환."""
    for rec_info in records[:max_try]:
        try:
            rec = wfdb.rdrecord(rec_info["record_path"])
        except Exception:
            continue
        sig = rec.p_signal if rec.p_signal is not None else rec.d_signal
        if sig is None:
            continue
        d         = rec.__dict__
        raw_names = [LEAD_ALIASES.get(n, n) for n in rec.sig_name]
        if (d["n_sig"] != 12
                or set(raw_names) != TARGET_SET
                or sig.shape[0] / d["fs"] < 1.0
                or has_zero_lead(sig)):
            continue
        return rec, sig, d, raw_names, rec_info
    return None


# ═══════════════════════════════════════════════════════════════
# 단일 데이터셋 테스트
# ═══════════════════════════════════════════════════════════════
def test_one(dataset_name: str, cfg: dict, output_root: Path, args) -> dict:
    prefix = cfg["prefix"]
    h5_dir = output_root / "data"
    os.makedirs(h5_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{dataset_name}]  prefix={prefix}")
    print(f"{'='*60}")

    # 1. 레코드 목록
    print("[1] 레코드 목록 로드")
    try:
        records = cfg["records_fn"]()
    except Exception as e:
        print(f"  [{FAIL}] {e}")
        return {"dataset": dataset_name, "status": FAIL, "reason": str(e)}
    if not records:
        print(f"  [{FAIL}] 레코드 없음 (경로 확인 필요)")
        return {"dataset": dataset_name, "status": FAIL, "reason": "no records"}
    print(f"  전체: {len(records):,}개")

    # 2. 유효 레코드 탐색
    print("[2] 유효 레코드 탐색 (최대 20개 시도)")
    found = _find_valid(records)
    if found is None:
        print(f"  [{FAIL}] 유효 레코드 없음")
        return {"dataset": dataset_name, "status": FAIL, "reason": "no valid record"}
    rec, sig, d, raw_names, rec_info = found

    fs      = d["fs"]
    sig_len = d["sig_len"]
    print(f"  record_path : {rec_info['record_path']}")
    print(f"  pid / rid   : {rec_info['pid']} / {rec_info['rid']}")
    print(f"  fs          : {fs} Hz  |  duration: {sig.shape[0]/fs:.1f}s")
    print(f"  leads       : {raw_names}")

    # 3. Reorder → (12, samples) fp16
    print("[3] Reorder → (12, samples) fp16")
    idx           = [raw_names.index(n) for n in TARGET_SIG_NAME]
    sig_reordered = sig[:, idx].T.astype(np.float16)
    sig_f32       = sig_reordered.astype(np.float32)
    nan_leads     = int(np.sum(np.all(np.isnan(sig_f32), axis=1)))
    print(f"  shape: {sig_reordered.shape}  NaN 리드: {nan_leads}/12")

    # 10초 세그먼트 분할
    SEG_SEC     = 10
    seg_samples = int(fs * SEG_SEC)
    n_segs      = max(1, sig_reordered.shape[1] // seg_samples)
    if n_segs > 1:
        trimmed = sig_reordered.shape[1] - n_segs * seg_samples
        print(f"  → {n_segs}개 세그먼트 분할 "
              f"({SEG_SEC}s × {n_segs}"
              + (f", 끝 {trimmed}샘플 버림)" if trimmed else ")"))
    sig_segs = [sig_reordered[:, i * seg_samples:(i + 1) * seg_samples]
                for i in range(n_segs)]

    # 4. signal_statistics
    print("[4] signal_statistics")
    try:
        stats = signal_statistics(sig_f32.T)
        print(f"  amp_std: {[round(v, 3) for v in stats['amp_std']]}")
    except Exception as e:
        print(f"  [{WARN}] {e}")

    # 5. beat_annotation (옵션) — 세그먼트별 계산
    ba_list, beat_method = None, ""
    if args.compute_beat:
        print("[5] beat_annotation")
        ba_list = []
        for seg in sig_segs:
            seg_f32 = seg.astype(np.float32)
            try:
                ba = extract_beat_annotation(np.nan_to_num(seg_f32[1]), fs)
            except Exception:
                ba = {"sample": [], "symbol": [], "subtype": [], "chan": [],
                      "num": [], "aux_note": []}
            ba_list.append(ba)
        beat_method  = "neurokit2"
        total_peaks  = sum(len(ba["sample"]) for ba in ba_list)
        print(f"  R-peaks: {total_peaks}개 ({n_segs}개 세그먼트)  [{PASS}]")

    # 6. fiducial (옵션) — 세그먼트별 계산
    fp_list, ff_list, fidu_method = None, None, ""
    if args.compute_fiducial:
        print("[6] fiducial")
        fp_list = []
        ff_list = []
        for seg in sig_segs:
            seg_f32 = seg.astype(np.float32)
            try:
                fp, ff = extract_fiducial(np.nan_to_num(seg_f32), fs)
            except Exception:
                fp = {"fsample": [], "fiducial": []}
                ff = {}
            fp_list.append(fp)
            ff_list.append(ff)
        fidu_method = "neurokit2-dwt"
        total_pts   = sum(len(fp["fsample"]) for fp in fp_list)
        nan_cnt     = sum(1 for v in (ff_list[0] if ff_list else {}).values()
                          if isinstance(v, (float, np.floating)) and np.isnan(v))
        print(f"  points: {total_pts}개 ({n_segs}개 세그먼트)  "
              f"feat NaN: {nan_cnt}/19  [{PASS}]")

    # age / gender (WFDB 헤더 보완)
    age    = rec_info.get("age",    -1.0)
    gender = rec_info.get("gender",  0)
    if rec_info.get("source") == "wfdb" and (age == -1.0 or gender == 0):
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
    file_name = f"{prefix}{pid}{rid}"
    h5_path   = h5_dir / f"{file_name}.h5"

    # 7. H5 저장
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
                signal              = sig_segs,
                seg_len             = n_segs,
                beat_annotation     = ba_list,
                fiducial_point      = fp_list,
                fiducial_feature    = ff_list,
            )
        print(f"  [{PASS}] 저장 완료")
    except Exception as e:
        print(f"  [{FAIL}] 저장 실패: {e}")
        return {"dataset": dataset_name, "status": FAIL, "reason": str(e)}

    # 8. H5 검증
    print("[8] H5 재로드 검증")
    errors = []
    sig_h5_shape = None
    try:
        with h5py.File(h5_path, "r") as f:
            fn_h5 = f.attrs.get("file_name", "")
            if fn_h5 != file_name:
                errors.append(f"file_name 불일치: {fn_h5}")
            print(f"  file_name  : {fn_h5}  {'[OK]' if fn_h5 == file_name else '[FAIL]'}")

            meta  = f["ECG/metadata"]
            fs_h5 = int(meta.attrs.get("fs", 0))
            sn    = [s.decode() if isinstance(s, bytes) else s for s in meta["sig_name"][()]]
            sn_ok = sn == TARGET_SIG_NAME
            if not sn_ok:
                errors.append(f"sig_name 불일치: {sn}")
            print(f"  fs         : {fs_h5}  sig_name: {'[OK]' if sn_ok else '[FAIL]'}")

            seg_grp_h5   = f["ECG/segments"]
            n_segs_h5    = int(seg_grp_h5.attrs.get("seg_len", 1))
            s0           = seg_grp_h5["0"]
            sig_h5       = s0["signal"][()]
            sig_h5_shape = sig_h5.shape
            if sig_h5.shape[0] != 12:
                errors.append(f"shape[0]={sig_h5.shape[0]} != 12")

            orig   = sig_segs[0].astype(np.float32)
            h5v    = sig_h5.astype(np.float32)
            mask   = ~(np.isnan(orig) | np.isnan(h5v))
            val_ok = np.allclose(orig[mask], h5v[mask], atol=0.01) if mask.any() else True
            if not val_ok:
                errors.append("signal 값 불일치")
            seg_info = f"  (세그먼트 {n_segs_h5}개)" if n_segs_h5 > 1 else ""
            print(f"  shape      : {sig_h5.shape}  dtype={sig_h5.dtype}  "
                  f"값일치: {'[OK]' if val_ok else '[FAIL]'}{seg_info}")

            if "beat_annotation" in s0:
                total_peaks_h5 = sum(
                    seg_grp_h5[str(i)]["beat_annotation/sample"].shape[0]
                    for i in range(n_segs_h5)
                    if "beat_annotation" in seg_grp_h5[str(i)]
                )
                print(f"  beat_annot : {total_peaks_h5}개 R-peak ({n_segs_h5}개 세그먼트)")
            if "fiducial_feature" in s0:
                nc = sum(1 for k in FIDUCIAL_FEATURE_KEYS
                         if np.isnan(float(s0["fiducial_feature"].attrs.get(k, float("nan")))))
                print(f"  fiducial   : NaN {nc}/19")
    except Exception as e:
        errors.append(f"재로드 실패: {e}")

    status = PASS if not errors else FAIL
    print(f"\n  [{status}] {dataset_name}")
    for err in errors:
        print(f"    [{FAIL}] {err}")

    # CSV row (convert_to_h5.process_one 출력과 동일 스키마)
    sid = 0
    oid = f"{prefix}{pid}{rid}{sid}"
    csv_row = {
        "filepath":      f"data/{file_name}.h5",
        "dataset":       dataset_name,
        "pid":           pid,
        "rid":           rid,
        "sid":           sid,
        "oid":           oid,
        "age":           age,
        "gender":        gender,
        "height":        np.nan,
        "weight":        np.nan,
        "fs":            fs,
        "channel_name":  str(TARGET_SIG_NAME),
        "_record_name":  d["record_name"],
        "_record_path":  rec_info["record_path"],
        "_h5_filename":  f"{file_name}.h5",
    }

    return {
        "dataset":   dataset_name,
        "status":    status,
        "file_name": file_name,
        "fs":        fs,
        "shape":     sig_h5_shape,
        "age":       age,
        "gender":    gender,
        "errors":    errors,
        "csv_row":   csv_row if status == PASS else None,
    }


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="ECG H5 변환 테스트 — 데이터셋별 첫 번째 유효 레코드 1개",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # physionet 8개
  python test_convert.py --group physionet \\
      --physionet_root /data/raw/physionet.org/files \\
      --output_root /tmp/h5_test

  # HEEDB
  python test_convert.py --group heedb \\
      --heedb_root /data/raw/heedb/ECG \\
      --output_root /tmp/h5_test

  # 전체
  python test_convert.py --group all \\
      --heedb_root ... --physionet_root ... --zzu_root ... \\
      --output_root /tmp/h5_test

  # 특정 데이터셋 + beat/fiducial
  python test_convert.py --dataset georgia,heedb_i0001 \\
      --physionet_root ... --heedb_root ... \\
      --output_root /tmp/h5_test \\
      --compute_beat --compute_fiducial
        """,
    )
    target_grp = parser.add_mutually_exclusive_group()
    target_grp.add_argument("--group",   type=str,
                            choices=["heedb", "physionet", "zzu", "all"],
                            help="그룹 단위 테스트")
    target_grp.add_argument("--dataset", type=str,
                            help="쉼표 구분 데이터셋명 (예: georgia,ptbxl,heedb_i0001)")

    parser.add_argument("--heedb_root",    type=str,
                        default="/home/irteam/ddn-opendata1/raw/heedb/ECG")
    parser.add_argument("--physionet_root", type=str,
                        default="/home/irteam/ddn-opendata1/raw/physionet.org/files")
    parser.add_argument("--zzu_root",       type=str,
                        default="/home/irteam/ddn-opendata1/raw/ZZU-pECG")
    parser.add_argument("--output_root",    type=str, default="/tmp/h5_test")
    parser.add_argument("--compute_beat",     action="store_true", help="beat_annotation 테스트")
    parser.add_argument("--compute_fiducial", action="store_true", help="fiducial 테스트")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    configs = _build_all_configs(args)
    # HEEDB는 metadata.csv가 수백만 행 → nrows=HEEDB_TEST_NROWS로 제한
    if args.heedb_root:
        _patch_heedb_configs(configs, args.heedb_root)

    if args.group == "heedb":
        target_datasets = HEEDB_DATASETS
    elif args.group == "physionet":
        target_datasets = PHYSIONET_DATASETS
    elif args.group == "zzu":
        target_datasets = ZZU_DATASETS
    elif args.group == "all":
        target_datasets = HEEDB_DATASETS + PHYSIONET_DATASETS + ZZU_DATASETS
    elif args.dataset:
        target_datasets = [d.strip() for d in args.dataset.split(",")]
        unknown = [d for d in target_datasets if d not in configs]
        if unknown:
            print(f"알 수 없는 데이터셋: {unknown}")
            print(f"선택 가능: {list(configs.keys())}")
            sys.exit(1)
    else:
        parser.print_help()
        return

    # 경로가 지정되지 않은 데이터셋 제외
    available = [d for d in target_datasets if d in configs]
    skipped   = [d for d in target_datasets if d not in configs]
    if skipped:
        print(f"\n[SKIP] 경로 미지정으로 건너뜀: {skipped}")

    results = []
    for ds_name in available:
        result = test_one(ds_name, configs[ds_name], output_root, args)
        results.append(result)

    # 최종 요약
    print(f"\n\n{'='*72}")
    print(f"  최종 요약")
    print(f"{'='*72}")
    header = f"  {'상태':<6}  {'데이터셋':<20}  {'file_name':<26}  {'fs':>5}  shape"
    print(header)
    print(f"  {'-'*68}")
    for r in results:
        icon = PASS if r["status"] == PASS else FAIL
        print(
            f"  [{icon}]  {r['dataset']:<20}  "
            f"{r.get('file_name', '-'):<26}  "
            f"{str(r.get('fs', '-')):>5}  "
            f"{str(r.get('shape', '-'))}"
        )
        for err in r.get("errors", []):
            print(f"           [{FAIL}] {err}")

    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_fail = len(results) - n_pass
    print(f"\n  PASS {n_pass} / FAIL {n_fail} / 전체 {len(results)}")
    print(f"  출력: {output_root}")
    print(f"{'='*72}\n")

    # ─── CSV 생성 + 컬럼 검증 ───
    csv_rows = [r["csv_row"] for r in results if r.get("csv_row") is not None]
    if csv_rows:
        print(f"{'='*72}")
        print(f"  CSV 생성 및 컬럼 검증")
        print(f"{'='*72}")
        df       = _make_table_df(csv_rows)
        csv_path = output_root / "ecg_table_test.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ecg_table_test.csv : {len(df):,}행 → {csv_path}")
        print(f"  컬럼 ({len(df.columns)}): {list(df.columns)}")

        missing = [c for c in TABLE_COLS if c not in df.columns]
        extra   = [c for c in df.columns  if c not in TABLE_COLS]
        leaked  = [c for c in df.columns  if c.startswith("_")]

        col_errs = []
        if missing: col_errs.append(f"필수 컬럼 누락: {missing}")
        if leaked:  col_errs.append(f"내부(_) 컬럼 누출: {leaked}")
        if list(df.columns)[:len(TABLE_COLS)] != TABLE_COLS:
            col_errs.append(f"TABLE_COLS 순서 불일치 (앞 {len(TABLE_COLS)}개 기준)")

        if col_errs:
            print(f"  [{FAIL}] 컬럼 검증 실패")
            for e in col_errs:
                print(f"    - {e}")
        else:
            print(f"  [{PASS}] 컬럼 검증 통과 (TABLE_COLS {len(TABLE_COLS)}개 일치"
                  + (f", 추가 컬럼 {extra}" if extra else "") + ")")
        print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
