import os
import numpy as np
import pandas as pd
import wfdb
import h5py

from create_h5_structure_heedb import create_h5_structure, TARGET_SIG_NAME
from utils_heedb import (
    reorder_signal, has_zero_lead,
    signal_statistics, beat_similarity,
    extract_beat_annotation, extract_fiducial,
)

# ============================================================
# 설정
# ============================================================
INSTITUTION = "I0001"
BASE_DIR = f"/home/irteam/opendata1/raw/heedb/ECG/{INSTITUTION}"
META_PATH = os.path.join(BASE_DIR, "metadata", "metadata.csv")
WFDB_ROOT = os.path.join(BASE_DIR, "WFDB")
OUTPUT_H5 = "/home/irteam/tykim/convert_h5/convert_raw_to_h5/test_record.h5"  # 테스트 출력 경로
PREFIX = "he1"

# ============================================================
# 1. metadata에서 1행 가져오기
# ============================================================
df = pd.read_csv(META_PATH, dtype=str, low_memory=False)
row = df.iloc[500]  # 아무 행이나
rid = 500
pid = str(row["BDSPPatientID"]).strip()
fn_raw = str(row["FileName"]).strip().lstrip("/")
fn_clean = fn_raw[5:] if fn_raw.startswith("WFDB/") else fn_raw

print(f"pid: {pid}, rid: {rid}")
print(f"FileName: {fn_raw}")
print(f"fn_clean: {fn_clean}")

# ============================================================
# 2. WFDB 로드
# ============================================================
rec = wfdb.rdrecord(os.path.join(WFDB_ROOT, fn_clean))
sig = rec.p_signal
d = rec.__dict__
fs = d["fs"]

print(f"\nrecord_name: {d['record_name']}")
print(f"fs: {fs}, sig_len: {d['sig_len']}, n_sig: {d['n_sig']}")
print(f"sig_name: {rec.sig_name}")
print(f"signal shape: {sig.shape}")
print(f"zero lead: {has_zero_lead(sig)}")

# ============================================================
# 3. reorder
# ============================================================
sig_reordered = reorder_signal(sig, rec.sig_name)  # (12, timepoints)
print(f"\nreordered shape: {sig_reordered.shape}")

reorder_idx = [rec.sig_name.index(n) for n in TARGET_SIG_NAME]
fmt_r = [d["fmt"][i] for i in reorder_idx]
gain_r = [d["adc_gain"][i] for i in reorder_idx]
bl_r = [d["baseline"][i] for i in reorder_idx]
units_r = [d["units"][i] for i in reorder_idx]
res_r = [d["adc_res"][i] for i in reorder_idx]
zero_r = [d["adc_zero"][i] for i in reorder_idx]

# ============================================================
# 4. beat_annotation + fiducial (전부 켜서 테스트)
# ============================================================
print("\nbeat_annotation 추출 중...")
ba = extract_beat_annotation(sig_reordered[1], fs)
print(f"  R-peaks: {len(ba['sample'])}개")

print("fiducial 추출 중...")
fp, ff = extract_fiducial(sig_reordered, fs)
print(f"  fiducial points: {len(fp['fsample'])}개")
print(f"  fiducial features: {ff}")

# ============================================================
# 5. signal_quality
# ============================================================
print("\nsignal_quality 계산 중...")
sq = signal_statistics(sig_reordered.T)
bs = beat_similarity(sig_reordered.T, sampling_rate=fs)
sq.update(bs)
print(f"  nan_ratio: {sq['nan_ratio']}")
print(f"  amp_std: {sq['amp_std']}")
print(f"  bs_corr: {sq['bs_corr']}")

# ============================================================
# 6. H5 저장
# ============================================================
file_name = f"{PREFIX}{pid}{rid}"
print(f"\nH5 저장: {OUTPUT_H5}")
print(f"  file_name: {file_name}")

with h5py.File(OUTPUT_H5, "w") as h5f:
    create_h5_structure(
        h5f,
        file_name=file_name,
        beat_ext_method="neurokit2",
        fidu_extract_method="neurokit2-dwt",
        record_name=d["record_name"],
        n_sig=12, fs=fs, sig_len=d["sig_len"],
        base_time=str(d.get("base_time", "")),
        base_date=str(d.get("base_date", "")),
        sig_name=TARGET_SIG_NAME,
        fmt=fmt_r, adc_gain=gain_r, baseline=bl_r,
        units=units_r, adc_res=res_r, adc_zero=zero_r,
        signal=[sig_reordered], seg_len=1,
        beat_annotation=[ba],
        fiducial_point=[fp],
        fiducial_feature=[ff],
    )

# ============================================================
# 7. 검증 — 저장된 H5 다시 읽어서 확인
# ============================================================
print(f"\n{'='*50}")
print(f"  검증")
print(f"{'='*50}")

with h5py.File(OUTPUT_H5, "r") as f:
    print("\n[root attrs]")
    for k in ["dataset_version", "file_name", "beat_ext_method", "fidu_extract_method"]:
        print(f"  {k}: {f.attrs[k]}")

    meta = f["ECG/metadata"]
    print("\n[metadata attrs]")
    for k in ["record_name", "n_sig", "fs", "sig_len", "base_time", "base_date", "dtype"]:
        print(f"  {k}: {meta.attrs[k]}")

    print("\n[metadata datasets]")
    sn = [s.decode() if isinstance(s, bytes) else s for s in meta["sig_name"][()]]
    print(f"  sig_name: {sn}")
    print(f"  sig_name 일치: {sn == TARGET_SIG_NAME}")

    seg = f["ECG/segments"]
    print(f"\n[segments] seg_len={seg.attrs['seg_len']}")

    s0 = seg["0"]
    sig_h5 = s0["signal"][()]
    print(f"\n[signal]")
    print(f"  shape: {sig_h5.shape}")
    print(f"  dtype: {sig_h5.dtype}")
    print(f"  원본과 일치: {np.allclose(sig_h5.astype(np.float32), sig_reordered.astype(np.float32), atol=0.01)}")
    print(f"  Lead I 처음 5개:  저장={sig_h5[0,:5]}, 원본={sig_reordered[0,:5]}")

    if "beat_annotation" in s0:
        ba_h5 = s0["beat_annotation"]
        print(f"\n[beat_annotation]")
        print(f"  sample: {ba_h5['sample'].shape}, first5={ba_h5['sample'][:5]}")
        print(f"  symbol: {ba_h5['symbol'].shape}, first5={ba_h5['symbol'][:5]}")
        print(f"  subtype 전부 0: {np.all(ba_h5['subtype'][()] == 0)}")

    if "fiducial_point" in s0:
        fp_h5 = s0["fiducial_point"]
        print(f"\n[fiducial_point]")
        print(f"  fsample: {fp_h5['fsample'].shape}")
        print(f"  fiducial: {fp_h5['fiducial'].shape}, first3={fp_h5['fiducial'][:3]}")

    if "fiducial_feature" in s0:
        ff_h5 = s0["fiducial_feature"]
        print(f"\n[fiducial_feature]")
        nan_cnt = 0
        for k in ["p_amp","q_amp","r_amp","s_amp","t_amp","qrs_dur","qt_int","rr_int","p_axis","r_axis","t_axis"]:
            v = ff_h5.attrs[k]
            if np.isnan(v): nan_cnt += 1
            print(f"  {k}: {v}")
        print(f"  NaN 수: {nan_cnt}/19")

# ============================================================
# 8. CSV row 미리보기
# ============================================================
age_raw = row.get("AgeAtAcquisition", None)
try:
    age = float(age_raw) / 365.25 / 100
except:
    age = -1

print(f"\n[CSV row 미리보기]")
print(f"  filepath: data/{file_name}.h5")
print(f"  pid: {pid}, rid: {rid}, sid: 0")
print(f"  oid: {PREFIX}{pid}{rid}0")
print(f"  age: {age:.6f} ({age_raw} days → {float(age_raw)/365.25:.1f} yr)")
print(f"  gender: {row.get('SexDSC', row.get('Sex', ''))}")
print(f"  fs: {fs}")

print(f"\n✅ 테스트 완료: {OUTPUT_H5}")