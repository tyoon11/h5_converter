# HEEDB H5 변환 파이프라인

HEEDB(Harvard Electrocardiography Database) 원본 WFDB 포맷을 내부 표준 H5 포맷으로 변환하는 파이프라인입니다.

---

## 파일 구성

```
heedb/
├── convert_to_h5_heedb.py       # 메인 변환 스크립트 (Ray 병렬)
├── create_h5_structure_heedb.py # H5 파일 구조 생성 함수
├── utils_heedb.py               # 신호 처리 유틸리티
├── test_convert.py              # 단일 레코드 변환 테스트
└── verify_h5.py                 # H5 파일 검증 스크립트
```

> `create_h5_structure_heedb.py` 와 `utils_heedb.py` 는 공개 데이터셋 파이프라인(`convert_to_h5_public.py` 등)과 공유합니다.

---

## 데이터셋 정보

| 기관 코드 | 출처 | 파일명 접두사 | 규모 |
|-----------|------|--------------|------|
| `I0001` | MGH (Massachusetts General Hospital) | `he1` | ~10.6M ECG |
| `I0006` | EUH (Emory University Hospital) | `he6` | ~1.0M ECG |

### 원본 데이터 경로 구조

```
/home/irteam/opendata1/raw/heedb/ECG/
├── I0001/
│   ├── metadata/metadata.csv      # BDSPPatientID, FileName, AgeAtAcquisition, SexDSC 등
│   ├── WFDB/                      # *.dat + *.hea (WFDB 포맷)
│   │   ├── S0001/1987/ ...
│   │   └── S0004/2019/ ...
│   ├── 12SL_diagnoses/
│   └── ICD_codes/
└── I0006/
    ├── metadata/metadata.csv      # BDSPPatientID, FileName, AgeAtAcquisition, Sex 등
    └── WFDB/
        └── 2010/ ... 2018/
```

---

## H5 파일 구조

```
{file_name}.h5
├── attrs
│   ├── dataset_version          # str   "1.0"
│   ├── file_name                # str   "he1{pid}{rid}"
│   ├── beat_ext_method          # str   "neurokit2" 또는 ""
│   └── fidu_extract_method      # str   "neurokit2-dwt" 또는 ""
└── ECG/
    ├── metadata/
    │   ├── attrs: record_name, n_sig(=12), fs, sig_len,
    │   │         base_time, base_date, dtype("fp16")
    │   └── datasets: sig_name, fmt, adc_gain, baseline,
    │                 units, adc_res, adc_zero
    └── segments/
        ├── attrs: seg_len
        └── 0/                   # 세그먼트 인덱스 (현재 단일 세그먼트)
            ├── signal           # float16, shape (12, timepoints)
            ├── beat_annotation/ # 선택 (--compute_beat)
            │   ├── sample       # int16, R-peak 샘플 인덱스
            │   ├── symbol       # UTF-8
            │   ├── subtype      # int16
            │   ├── chan         # int16
            │   ├── num          # int16
            │   └── aux_note     # UTF-8
            ├── fiducial_point/  # 선택 (--compute_fiducial)
            │   ├── fsample      # int16, 샘플 인덱스
            │   └── fiducial     # UTF-8, 파형 레이블
            └── fiducial_feature/ # 선택 (--compute_fiducial)
                └── attrs: p_amp, q_amp, r_amp, s_amp, t_amp,
                           p_dur, pr_seg, qrs_dur, st_seg, t_dur,
                           pr_int, qt_int, rr_int, tp_seg,
                           qtc_baz, qtc_frid, p_axis, r_axis, t_axis
```

### 리드 순서

모든 H5 파일의 `signal`은 아래 순서로 고정됩니다.

```python
TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']
#  index           [ 0,   1,    2,    3,   4,   5,   6,   7,   8,    9,   10,   11 ]
```

---

## 출력 파일

```
/home/irteam/opendata1/h5/heedb/v4.0/
├── data/
│   ├── he1{pid}{rid}.h5         # I0001 H5 파일들
│   └── he6{pid}{rid}.h5         # I0006 H5 파일들
├── heedb_table.csv              # 학습용 메타 테이블
├── file_name.csv                # 원본 ↔ H5 파일명 매핑
├── combined_metadata.csv        # 양 기관 원본 메타데이터 통합
└── conversion.log
```

### `heedb_table.csv` 컬럼

| 컬럼 | 시점 | 설명 |
|------|------|------|
| `filepath` | 변환 시 | `data/{file_name}.h5` 상대 경로 |
| `pid` | 변환 시 | BDSPPatientID |
| `rid` | 변환 시 | metadata.csv 행 인덱스 |
| `sid` | 변환 시 | 세그먼트 인덱스 (항상 0) |
| `oid` | 변환 시 | 고유 관측 ID (`{prefix}{pid}{rid}{sid}`) |
| `age` | 변환 시 | 나이 (AgeAtAcquisition 일 수 → years / 100, 0~1 범위) |
| `gender` | 변환 시 | 1=남성, -1=여성, 0=미상 |
| `height` | 변환 시 | NaN (미제공) |
| `weight` | 변환 시 | NaN (미제공) |
| `fs` | 변환 시 | 샘플링 주파수 (Hz) |
| `channel_name` | 변환 시 | TARGET_SIG_NAME 리스트 문자열 |
| `nan_ratio` | **변환 시** | per-lead NaN 비율 |
| `amp_mean` | **변환 시** | per-lead 진폭 평균 |
| `amp_std` | **변환 시** | per-lead 진폭 표준편차 |
| `amp_skewness` | **변환 시** | per-lead 왜도 |
| `amp_kurtosis` | **변환 시** | per-lead 첨도 |
| `bs_corr` | **별도 계산** | per-lead beat-to-beat 상관계수 |
| `bs_dtw` | **별도 계산** | per-lead beat-to-beat DTW 거리 |

---

## 사용법

### 0. 환경 준비

```bash
pip install wfdb h5py ray tqdm numpy pandas neurokit2 dtw-python scipy
```

### 1. 단일 레코드 테스트

```bash
# 프로젝트 루트에서 실행
python heedb/test_convert.py
```

`metadata.csv` 500번째 행을 가져와 변환 → H5 저장 → 재로드 검증까지 수행합니다.

### 2. 전체 변환

변환 시 `signal_statistics` (nan_ratio, amp_*)를 **기본으로 함께 계산**합니다.

```bash
# 신호 저장 + signal_statistics (기본)
python heedb/convert_to_h5_heedb.py --num_cpus 64

# beat annotation 포함
python heedb/convert_to_h5_heedb.py --num_cpus 64 --compute_beat

# beat + fiducial 포함
python heedb/convert_to_h5_heedb.py --num_cpus 64 --compute_beat --compute_fiducial

# signal_statistics 생략 (변환만)
python heedb/convert_to_h5_heedb.py --num_cpus 64 --no_stats
```

### 3. 검증

```bash
# 단일 파일 상세 검증
python heedb/verify_h5.py --file /home/irteam/opendata1/h5/heedb/v4.0/data/he1123450.h5

# 폴더 전체 샘플 검증
python heedb/verify_h5.py --dir /home/irteam/opendata1/h5/heedb/v4.0/data --sample 200
```

### 4. beat_similarity 계산 (별도)

`bs_corr` / `bs_dtw` 는 계산 비용이 크므로 변환 완료 후 따로 실행합니다.

```bash
# 프로젝트 루트에서 실행
python append_signal_quality.py \
    --csv  /home/irteam/opendata1/h5/heedb/v4.0/heedb_table.csv \
    --h5_root /home/irteam/opendata1/h5/heedb/v4.0 \
    --num_cpus 64 \
    --backup

# DTW 생략 (bs_dtw = NaN, 속도 우선)
python append_signal_quality.py \
    --csv  /home/irteam/opendata1/h5/heedb/v4.0/heedb_table.csv \
    --h5_root /home/irteam/opendata1/h5/heedb/v4.0 \
    --no_dtw --num_cpus 64
```

`append_signal_quality.py`는 `bs_corr` 컬럼이 비어있는 행만 처리하므로 중단 후 재실행해도 안전합니다.

---

## 스킵 조건

| 조건 | 이유 |
|------|------|
| `BDSPPatientID` 결측 또는 `"nan"` | 식별 불가 |
| WFDB 로드 실패 | 파일 손상 또는 누락 |
| `n_sig ≠ 12` | 12리드가 아닌 레코드 |
| 신호 길이 < 1초 | 너무 짧음 |
| 리드 집합이 TARGET_SIG_NAME과 불일치 | 비표준 리드 구성 |
| 전체 0인 리드 존재 | 비정상 신호 |
| 이미 H5 파일 존재 | 증분 변환 (중복 방지) |

---

## 모듈 설명

### `create_h5_structure_heedb.py`

H5 파일 내부 구조를 생성하는 `create_h5_structure()` 함수를 제공합니다.
HEEDB와 공개 데이터셋 파이프라인이 이 함수를 공유합니다.

### `utils_heedb.py`

| 함수 | 입력 | 출력 | 설명 |
|------|------|------|------|
| `reorder_signal(signal, actual_sig_name)` | `(T, N)`, `list[str]` | `(12, T) float16` | WFDB 리드 순서 → TARGET 순서 재정렬 |
| `has_zero_lead(signal)` | `(T, N)` | `bool` | 전체 0인 리드 존재 여부 |
| `extract_beat_annotation(signal_lead2, fs)` | `1D array`, `int` | `dict` | neurokit2 R-peak 검출 |
| `extract_fiducial(signal_reordered, fs)` | `(12, T)`, `int` | `(dict, dict)` | P/Q/R/S/T 파형 위치 + 측정값 |
| `signal_statistics(signal)` | `(T, 12)` | `dict` | per-lead NaN 비율, 진폭 통계 — **변환 시 기본 계산** |
| `beat_similarity(signal, fs)` | `(T, 12)`, `int` | `dict` | per-lead beat-to-beat 상관계수 + DTW — **별도 계산** |

---

## CLI 옵션 (`convert_to_h5_heedb.py`)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--num_cpus` | `64` | Ray 병렬 CPU 수 |
| `--batch_size` | `5000` | 배치 크기 |
| `--compute_beat` | OFF | beat_annotation 생성 |
| `--compute_fiducial` | OFF | fiducial_point / feature 생성 |
| `--no_stats` | OFF | signal_statistics 계산 생략 (기본은 계산) |