"""
H5 구조 생성 함수
==================
모든 데이터셋(HEEDB, PhysioNet, ZZU)이 공유하는 표준 H5 파일 구조를 생성합니다.
"""

import h5py
import numpy as np

UTF8 = h5py.string_dtype(encoding="utf-8")

TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']

FIDUCIAL_FEATURE_KEYS = [
    "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
    "p_dur", "pr_seg", "qrs_dur", "st_seg", "t_dur",
    "pr_int", "qt_int", "rr_int", "tp_seg",
    "qtc_baz", "qtc_frid",
    "p_axis", "r_axis", "t_axis",
]


def create_h5_structure(
    h5_file,
    file_name, beat_ext_method="", fidu_extract_method="", dataset_version="1.0",
    record_name="", n_sig=12, fs=500, sig_len=5000,
    base_time="", base_date="",
    sig_name=None, fmt=None, adc_gain=None, baseline=None,
    units=None, adc_res=None, adc_zero=None,
    signal=None, seg_len=1,
    beat_annotation=None, fiducial_point=None, fiducial_feature=None,
):
    """
    표준 H5 파일 구조를 생성합니다.

    구조:
      root attrs: dataset_version, file_name, beat_ext_method, fidu_extract_method
      ECG/
        metadata/  (attrs + datasets)
        segments/
          0/
            signal            (12, T) float16
            beat_annotation/  (선택)
            fiducial_point/   (선택)
            fiducial_feature/ (선택)
    """
    # root attrs
    h5_file.attrs["dataset_version"] = dataset_version
    h5_file.attrs["file_name"] = file_name
    h5_file.attrs["beat_ext_method"] = beat_ext_method
    h5_file.attrs["fidu_extract_method"] = fidu_extract_method

    ecg_grp = h5_file.create_group("ECG")

    # metadata
    meta_grp = ecg_grp.create_group("metadata")
    meta_grp.attrs["record_name"] = record_name
    meta_grp.attrs["n_sig"] = n_sig
    meta_grp.attrs["fs"] = fs
    meta_grp.attrs["sig_len"] = sig_len
    meta_grp.attrs["base_time"] = base_time if base_time is not None else ""
    meta_grp.attrs["base_date"] = base_date if base_date is not None else ""
    meta_grp.attrs["dtype"] = "fp16"

    _sn = sig_name if sig_name is not None else TARGET_SIG_NAME
    meta_grp.create_dataset("sig_name", data=np.array(_sn, dtype=UTF8), dtype=UTF8)
    if fmt is not None:
        meta_grp.create_dataset("fmt", data=np.array(fmt, dtype=UTF8), dtype=UTF8)
    if adc_gain is not None:
        meta_grp.create_dataset("adc_gain", data=np.array(adc_gain, dtype=np.float16))
    if baseline is not None:
        meta_grp.create_dataset("baseline", data=np.array(baseline, dtype=np.int16))
    if units is not None:
        meta_grp.create_dataset("units", data=np.array(units, dtype=UTF8), dtype=UTF8)
    if adc_res is not None:
        meta_grp.create_dataset("adc_res", data=np.array(adc_res, dtype=np.int16))
    if adc_zero is not None:
        meta_grp.create_dataset("adc_zero", data=np.array(adc_zero, dtype=np.int16))

    # segments
    seg_grp = ecg_grp.create_group("segments")
    seg_grp.attrs["seg_len"] = seg_len

    for i in range(seg_len):
        s_grp = seg_grp.create_group(str(i))

        if signal is not None and i < len(signal):
            s_grp.create_dataset("signal", data=np.array(signal[i], dtype=np.float16))

        if beat_annotation is not None and i < len(beat_annotation):
            ba = beat_annotation[i]
            ba_grp = s_grp.create_group("beat_annotation")
            samples = np.array(ba.get("sample", []), dtype=np.int16)
            nb = len(samples)
            ba_grp.create_dataset("sample",   data=samples)
            ba_grp.create_dataset("symbol",   data=np.array(ba.get("symbol",   [""]*nb), dtype=UTF8), dtype=UTF8)
            ba_grp.create_dataset("subtype",  data=np.array(ba.get("subtype",  np.zeros(nb)), dtype=np.int16))
            ba_grp.create_dataset("chan",      data=np.array(ba.get("chan",     np.zeros(nb)), dtype=np.int16))
            ba_grp.create_dataset("num",       data=np.array(ba.get("num",      np.zeros(nb)), dtype=np.int16))
            ba_grp.create_dataset("aux_note",  data=np.array(ba.get("aux_note", [""]*nb), dtype=UTF8), dtype=UTF8)

        if fiducial_point is not None and i < len(fiducial_point):
            fp = fiducial_point[i]
            fp_grp  = s_grp.create_group("fiducial_point")
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

        if fiducial_feature is not None and i < len(fiducial_feature):
            ff = fiducial_feature[i]
            ff_grp = s_grp.create_group("fiducial_feature")
            for key in FIDUCIAL_FEATURE_KEYS:
                val = ff.get(key, np.nan)
                try:
                    ff_grp.attrs[key] = np.float16(val)
                except (TypeError, ValueError):
                    ff_grp.attrs[key] = np.float16(np.nan)
