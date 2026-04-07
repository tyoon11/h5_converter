"""
신호 처리 유틸리티
==================
모든 데이터셋이 공유하는 신호 처리 함수들입니다.

Functions:
  reorder_signal       : WFDB sig_name 순서 → TARGET_SIG_NAME 순서로 재배열
  has_zero_lead        : 전체 0인 리드 존재 여부 확인
  signal_statistics    : per-lead 신호 통계 (nan_ratio, amp_mean/std/skewness/kurtosis)
  beat_similarity      : per-lead beat 유사도 (bs_corr, bs_dtw)
  extract_beat_annotation : neurokit2 R-peak 검출
  extract_fiducial     : neurokit2 fiducial point + 19개 feature 추출
"""

import numpy as np
import warnings

TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════
# signal reorder
# ═══════════════════════════════════════════════════════════════
def reorder_signal(signal, actual_sig_name):
    """
    WFDB 원본 순서의 신호를 TARGET_SIG_NAME 순서로 재배열하고 전치합니다.

    Args:
        signal: (samples, n_leads) — WFDB 원본 순서
        actual_sig_name: record.sig_name 리스트
    Returns:
        (n_leads, samples) float16 — TARGET_SIG_NAME 순서
    """
    idx = [actual_sig_name.index(n) for n in TARGET_SIG_NAME]
    reordered = signal[:, idx]  # (samples, 12) reordered
    return reordered.T          # (12, samples)


def has_zero_lead(signal):
    """
    전체가 0인 리드가 1개라도 있으면 True를 반환합니다.

    Args:
        signal: (samples, n_leads) — WFDB 원본 순서
    """
    return np.any(np.all(signal == 0, axis=0))


# ═══════════════════════════════════════════════════════════════
# signal_statistics (CSV 컬럼용)
# ═══════════════════════════════════════════════════════════════
def signal_statistics(signal):
    """
    per-lead 신호 통계를 계산합니다.

    Args:
        signal: (samples, n_leads)
    Returns:
        dict: nan_ratio, amp_mean, amp_std, amp_skewness, amp_kurtosis (각 12-element list)
    """
    from scipy.stats import skew, kurtosis

    sig = np.array(signal, dtype=np.float32)
    n_leads = sig.shape[1]

    nan_ratio = np.mean(np.isnan(sig), axis=0)
    amp_mean  = np.array([np.nanmean(sig[:, i]) if not np.all(np.isnan(sig[:, i])) else 0.0 for i in range(n_leads)])
    amp_std   = np.array([np.nanstd(sig[:, i])  if not np.all(np.isnan(sig[:, i])) else 0.0 for i in range(n_leads)])
    amp_skew  = np.array([skew(sig[:, i],     nan_policy="omit") if not np.all(np.isnan(sig[:, i])) else 0.0 for i in range(n_leads)])
    amp_kurt  = np.array([kurtosis(sig[:, i], nan_policy="omit") if not np.all(np.isnan(sig[:, i])) else 0.0 for i in range(n_leads)])

    return {
        "nan_ratio":    list(np.nan_to_num(nan_ratio, nan=0.0)),
        "amp_mean":     list(np.nan_to_num(amp_mean,  nan=0.0)),
        "amp_std":      list(np.nan_to_num(amp_std,   nan=0.0)),
        "amp_skewness": list(np.nan_to_num(amp_skew,  nan=0.0)),
        "amp_kurtosis": list(np.nan_to_num(amp_kurt,  nan=0.0)),
    }


def beat_similarity(signal, sampling_rate=500):
    """
    per-lead beat 유사도를 계산합니다 (beat-to-beat correlation + DTW distance).

    Args:
        signal: (samples, n_leads)
        sampling_rate: 샘플링 주파수
    Returns:
        dict: bs_corr, bs_dtw (각 12-element list)
    """
    try:
        import neurokit2 as nk
        from dtw import dtw
    except ImportError:
        n = signal.shape[1]
        return {"bs_corr": [np.nan] * n, "bs_dtw": [np.nan] * n}

    n_leads      = signal.shape[1]
    fixed_length = sampling_rate * 2
    mean_corrs   = [np.nan] * n_leads
    mean_dtws    = [np.nan] * n_leads

    for idx in range(n_leads):
        try:
            _, rpeaks = nk.ecg_peaks(signal[:, idx], sampling_rate=sampling_rate)
            rpeaks    = rpeaks.get("ECG_R_Peaks", [])
            if len(rpeaks) <= 3:
                continue

            beats = nk.ecg_segment(signal[:, idx], rpeaks, sampling_rate=sampling_rate)
            if len(beats) <= 3:
                continue

            beat_matrix = []
            for beat in beats.values():
                br  = nk.signal_resample(np.array(beat, dtype=float), desired_length=fixed_length)
                std = np.std(br)
                if std == 0 or np.isnan(std):
                    br = np.zeros_like(br)
                else:
                    br = (br - np.mean(br)) / std
                beat_matrix.append(br.squeeze())
            beat_matrix = np.array(beat_matrix)

            # beat-to-beat correlation
            corrs = []
            for i in range(len(beat_matrix) - 1):
                if not np.any(np.isnan(beat_matrix[i])) and not np.any(np.isnan(beat_matrix[i + 1])):
                    c = np.corrcoef(beat_matrix[i], beat_matrix[i + 1])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
            mean_corrs[idx] = np.mean(corrs) if corrs else np.nan

            # DTW distance
            dtw_dists = []
            for i in range(len(beat_matrix) - 1):
                if np.any(np.isnan(beat_matrix[i])) or np.any(np.isnan(beat_matrix[i + 1])):
                    continue
                try:
                    alignment = dtw(beat_matrix[i], beat_matrix[i + 1])
                    dtw_dists.append(alignment.distance / fixed_length)
                except Exception:
                    pass
            mean_dtws[idx] = np.nanmean(dtw_dists) if dtw_dists else np.nan
        except Exception:
            continue

    return {"bs_corr": mean_corrs, "bs_dtw": mean_dtws}


# ═══════════════════════════════════════════════════════════════
# beat_annotation (H5 선택 저장)
# ═══════════════════════════════════════════════════════════════
def extract_beat_annotation(signal_lead2, sampling_rate):
    """
    Lead II 신호에서 R-peak를 검출하여 beat_annotation 딕셔너리를 반환합니다.

    Args:
        signal_lead2: 1D array (Lead II)
        sampling_rate: 샘플링 주파수
    Returns:
        dict: sample, symbol, subtype, chan, num, aux_note
    """
    import neurokit2 as nk

    try:
        cleaned = nk.ecg_clean(signal_lead2, sampling_rate=sampling_rate)
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        peaks = rpeaks.get("ECG_R_Peaks", [])
    except Exception:
        peaks = []

    n = len(peaks)
    return {
        "sample":   list(peaks),
        "symbol":   [""] * n,
        "subtype":  [0]  * n,
        "chan":      [0]  * n,
        "num":       [0]  * n,
        "aux_note": [""] * n,
    }


# ═══════════════════════════════════════════════════════════════
# fiducial_point + fiducial_feature (H5 선택 저장)
# ═══════════════════════════════════════════════════════════════
def extract_fiducial(signal_reordered, sampling_rate):
    """
    12-lead 신호에서 fiducial point와 19개 feature를 추출합니다.

    Args:
        signal_reordered: (12, timepoints) — TARGET_SIG_NAME 순서
        sampling_rate: 샘플링 주파수
    Returns:
        (fiducial_point_dict, fiducial_feature_dict)
    """
    import neurokit2 as nk

    feature_keys = [
        "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
        "p_dur", "pr_seg", "qrs_dur", "st_seg", "t_dur",
        "pr_int", "qt_int", "rr_int", "tp_seg",
        "qtc_baz", "qtc_frid", "p_axis", "r_axis", "t_axis",
    ]
    fidu_point = {"fsample": [], "fiducial": []}
    fidu_feat  = {k: np.nan for k in feature_keys}

    sig_ii = signal_reordered[1]
    sig_i  = signal_reordered[0]

    try:
        cleaned_ii = nk.ecg_clean(sig_ii, sampling_rate=sampling_rate)
        _, rpeaks  = nk.ecg_peaks(cleaned_ii, sampling_rate=sampling_rate)
        rpeaks     = rpeaks.get("ECG_R_Peaks", [])

        if len(rpeaks) <= 3:
            return fidu_point, fidu_feat

        _, waves = nk.ecg_delineate(cleaned_ii, rpeaks, sampling_rate=sampling_rate, method="dwt", show=False)
        waves["ECG_R_Peaks"] = rpeaks

        fidu_names = [
            "ECG_P_Onsets", "ECG_P_Peaks", "ECG_Q_Onsets", "ECG_P_Offsets",
            "ECG_Q_Peaks", "ECG_R_Peaks", "ECG_S_Peaks", "ECG_R_Offsets",
            "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets",
        ]
        fs_list, fid_list = [], []
        for label in fidu_names:
            pts = waves.get(label)
            if pts is not None:
                pts   = np.array(pts, dtype=np.float32)
                valid = ~np.isnan(pts)
                pts   = pts[valid].astype(int)
                fs_list.extend(pts.tolist())
                fid_list.extend([label] * len(pts))

        if fs_list:
            order = np.argsort(fs_list)
            fidu_point["fsample"]  = [fs_list[i]  for i in order]
            fidu_point["fiducial"] = [fid_list[i] for i in order]

        def _interval(a, b):
            a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
            if len(a) != len(b) or len(a) <= 3:
                return np.nan
            valid = ~np.isnan(a) & ~np.isnan(b)
            return np.nanmean(b[valid] - a[valid]) if np.any(valid) else np.nan

        def _amp(sig, peaks, refs):
            peaks, refs = np.array(peaks, dtype=np.float32), np.array(refs, dtype=np.float32)
            if len(peaks) != len(refs) or len(peaks) <= 3:
                return np.nan
            valid = ~np.isnan(peaks) & ~np.isnan(refs)
            if not np.any(valid):
                return np.nan
            return np.nanmean(sig[peaks[valid].astype(int)] - sig[refs[valid].astype(int)])

        fidu_feat["p_amp"] = _amp(cleaned_ii, waves.get("ECG_P_Peaks", []), waves.get("ECG_P_Onsets", []))
        fidu_feat["q_amp"] = _amp(cleaned_ii, waves.get("ECG_Q_Peaks", []), waves.get("ECG_P_Onsets", []))
        fidu_feat["r_amp"] = _amp(cleaned_ii, waves.get("ECG_R_Peaks", []), waves.get("ECG_Q_Peaks", []))
        fidu_feat["s_amp"] = _amp(cleaned_ii, waves.get("ECG_S_Peaks", []), waves.get("ECG_P_Onsets", []))
        fidu_feat["t_amp"] = _amp(cleaned_ii, waves.get("ECG_T_Peaks", []), waves.get("ECG_T_Onsets", []))

        fs_rate = sampling_rate
        fidu_feat["p_dur"]   = _interval(waves.get("ECG_P_Onsets",  []), waves.get("ECG_P_Offsets", [])) / fs_rate
        fidu_feat["pr_seg"]  = _interval(waves.get("ECG_P_Offsets", []), waves.get("ECG_Q_Peaks",   [])) / fs_rate
        fidu_feat["qrs_dur"] = _interval(waves.get("ECG_Q_Peaks",   []), waves.get("ECG_S_Peaks",   [])) / fs_rate
        fidu_feat["st_seg"]  = _interval(waves.get("ECG_S_Peaks",   []), waves.get("ECG_T_Onsets",  [])) / fs_rate
        fidu_feat["t_dur"]   = _interval(waves.get("ECG_T_Onsets",  []), waves.get("ECG_T_Offsets", [])) / fs_rate
        fidu_feat["pr_int"]  = _interval(waves.get("ECG_P_Onsets",  []), waves.get("ECG_Q_Peaks",   [])) / fs_rate
        fidu_feat["qt_int"]  = _interval(waves.get("ECG_Q_Peaks",   []), waves.get("ECG_T_Offsets", [])) / fs_rate
        fidu_feat["rr_int"]  = _interval(rpeaks[:-1], rpeaks[1:]) / fs_rate
        fidu_feat["tp_seg"]  = _interval(waves.get("ECG_T_Offsets", [None])[:-1], waves.get("ECG_P_Onsets", [None])[1:]) / fs_rate

        if isinstance(fidu_feat["rr_int"], float) and fidu_feat["rr_int"] > 0:
            fidu_feat["qtc_baz"]  = fidu_feat["qt_int"] / np.sqrt(fidu_feat["rr_int"])
            fidu_feat["qtc_frid"] = fidu_feat["qt_int"] / fidu_feat["rr_int"] ** (1 / 3)

        cleaned_i = nk.ecg_clean(sig_i, sampling_rate=sampling_rate)
        for name, pts_key in [("p_axis", "ECG_P_Peaks"), ("r_axis", "ECG_R_Peaks"), ("t_axis", "ECG_T_Peaks")]:
            pts = [p for p in waves.get(pts_key, []) if isinstance(p, (int, np.integer))]
            if pts:
                fidu_feat[name] = float(np.nanmean(np.degrees(np.arctan2(cleaned_ii[pts], cleaned_i[pts]))))

        fidu_feat = {
            k: np.float16(v) if isinstance(v, (float, int, np.number)) else np.float16(np.nan)
            for k, v in fidu_feat.items()
        }

    except Exception:
        pass

    return fidu_point, fidu_feat
