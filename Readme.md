# HEEDB H5 Converter

WFDB 포맷의 HEEDB 원본 ECG 데이터를 HDF5(`.h5`) 파일로 변환하는 파이프라인입니다.  
I0001(MGH)과 I0006(EUH) 두 기관의 데이터를 병렬로 처리하며, 변환 결과와 함께 학습용 CSV 테이블을 자동으로 생성합니다.

---

## 디렉토리 구조

```
convert_raw_to_h5/
├── convert_to_h5_heedb.py        # 메인 변환 스크립트 (Ray 병렬 처리)
├── create_h5_structure_heedb.py  # H5 내부 구조 정의 및 저장 함수
├── utils_heedb.py                # 신호 처리 유틸리티 (피듀셜, 품질 지표 등)
├── verify_h5.py                  # 변환 결과 검증 스크립트
└── test_convert.py               # 단일 레코드 변환 테스트
```

### 원본 데이터 구조 (변환 전)

```
/home/irteam/opendata1/raw/heedb/ECG/
├── I0001/
│   ├── metadata/
│   │   └── metadata.csv
│   └── WFDB/
│       ├── S0001/{연도}/...   ← *.dat + *.hea
│       ├── S0002/{연도}/...
│       └── ...
└── I0006/
    ├── metadata/
    │   └── metadata.csv
    └── WFDB/
        ├── {연도}/...        ← *.dat + *.hea
        └── ...
```

### 변환 결과 구조

```
/home/irteam/opendata1/h5/heedb/v4.0/
├── data/
│   ├── he1{pid}{rid}.h5
│   ├── he6{pid}{rid}.h5
│   └── ...
├── heedb_table.csv       # 학습용 메인 테이블
├── file_name.csv         # 원본↔H5 파일 경로 매핑
├── combined_metadata.csv # I0001 + I0006 통합 원본 메타데이터
└── conversion.log        # 변환 로그
```

---

## 의존성 설치

```bash
pip install h5py wfdb numpy pandas ray tqdm neurokit2 scipy dtw-python
```

---

## 빠른 시작

### 1. 기본 실행 (신호 품질 지표만 포함)

```bash
python convert_to_h5_heedb.py
```

### 2. 옵션 지정 실행

```bash
# CPU 수 조정
python convert_to_h5_heedb.py --num_cpus 32

# R-peak 어노테이션 포함
python convert_to_h5_heedb.py --compute_beat

# 피듀셜 포인트/피처 포함 (시간 오래 걸림)
python convert_to_h5_heedb.py --compute_fiducial

# R-peak + 피듀셜 모두 포함
python convert_to_h5_heedb.py --compute_beat --compute_fiducial

# 신호 품질 지표 제외 (속도 우선)
python convert_to_h5_heedb.py --no_quality
```

---

## 실행 옵션 전체 목록

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--num_cpus` | `64` | Ray 워커 CPU 수 |
| `--batch_size` | `5000` | 배치당 레코드 수 |
| `--compute_beat` | `False` | R-peak 어노테이션 생성 (NeuroKit2) |
| `--compute_fiducial` | `False` | 피듀셜 포인트/피처 생성 (NeuroKit2 DWT) |
| `--compute_quality` | `True` | 신호 품질 지표 생성 |
| `--no_quality` | — | `--compute_quality` 비활성화 플래그 |

---

## 변환 파이프라인 상세

```
metadata.csv 로드
      │
      ▼
레코드별 처리 (@ray.remote, 병렬)
      │
      ├─ 1. WFDB 로드 (wfdb.rdrecord)
      │
      ├─ 2. 스킵 조건 검사
      │     ├ pid 결측 → skip
      │     ├ n_sig ≠ 12 → skip
      │     ├ duration < 1s → skip
      │     ├ 채널 집합 불일치 → skip
      │     └ zero-lead 존재 → skip
      │
      ├─ 3. 신호 reorder + transpose
      │     WFDB (samples, n_sig)
      │       → 채널 순서 고정 [I,II,III,V1..V6,aVF,aVL,aVR]
      │       → (12, samples) float16
      │
      ├─ 4. [옵션] beat_annotation 추출
      │     extract_beat_annotation(Lead II, fs)
      │     → NeuroKit2 R-peak 검출
      │
      ├─ 5. [옵션] fiducial 추출
      │     extract_fiducial(signal_reordered, fs)
      │     → Lead II DWT 파형 분할
      │     → 피듀셜 포인트 + 19개 피처
      │
      ├─ 6. [옵션] 신호 품질 계산
      │     signal_statistics → NaN 비율, 진폭 통계 (5종)
      │     beat_similarity   → 박동 간 상관계수, DTW 거리
      │
      └─ 7. H5 저장 + CSV row 반환
            create_h5_structure(...)
            → data/{file_name}.h5

배치 완료 후
      │
      ├─ heedb_table.csv 저장
      ├─ file_name.csv 저장
      └─ combined_metadata.csv 저장
```

---

## 모듈별 설명

### `convert_to_h5_heedb.py`

메인 스크립트. Ray를 이용한 병렬 변환을 수행합니다.

**주요 동작:**
- I0001, I0006 순서대로 각 기관의 `metadata.csv` 를 읽어 전체 레코드를 처리합니다.
- 이미 변환된 파일(`.h5` 존재)은 자동으로 건너뜁니다 — **중단 후 재시작 시 이어서 처리됩니다.**
- `batch_size` 단위로 나누어 Ray future를 관리합니다.

**기관 설정:**

```python
INSTITUTIONS = [
    {
        "name": "I0001",
        "prefix": "he1",
        "base_dir": "/home/irteam/opendata1/raw/heedb/ECG/I0001",
        "gender_field": "SexDSC",     # I0001은 "SexDSC" 컬럼 사용
    },
    {
        "name": "I0006",
        "prefix": "he6",
        "base_dir": "/home/irteam/opendata1/raw/heedb/ECG/I0006",
        "gender_field": "Sex",        # I0006은 "Sex" 컬럼 사용
    },
]
```

---

### `create_h5_structure_heedb.py`

H5 파일 내부 구조를 정의하고 데이터를 저장하는 함수 `create_h5_structure()` 를 제공합니다.

```python
from create_h5_structure_heedb import create_h5_structure

with h5py.File("output.h5", "w") as h5f:
    create_h5_structure(
        h5f,
        file_name="he1{pid}{rid}",
        beat_ext_method="neurokit2",          # "" 이면 beat_annotation 미저장
        fidu_extract_method="neurokit2-dwt",  # "" 이면 fiducial 미저장
        record_name="...",
        n_sig=12, fs=500, sig_len=5000,
        base_time="...", base_date="...",
        sig_name=['I','II','III','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR'],
        fmt=[...], adc_gain=[...], baseline=[...],
        units=[...], adc_res=[...], adc_zero=[...],
        signal=[sig_reordered],    # list of (12, samples) arrays
        seg_len=1,
        beat_annotation=[ba],      # None이면 미저장
        fiducial_point=[fp],       # None이면 미저장
        fiducial_feature=[ff],     # None이면 미저장
    )
```

---

### `utils_heedb.py`

신호 처리 핵심 함수 모음입니다.

#### `reorder_signal(signal, actual_sig_name)`

WFDB 로드 후 채널 순서를 고정 순서로 재배열합니다.

```python
# signal: (samples, n_leads) ← WFDB 기본 shape
# 반환: (12, samples) float16 ← H5 저장용
sig_reordered = reorder_signal(rec.p_signal, rec.sig_name)
```

#### `has_zero_lead(signal)`

전체 값이 0인 채널이 있으면 `True` 를 반환합니다. 변환 스킵 조건으로 사용됩니다.

```python
# signal: (samples, n_leads)
if has_zero_lead(sig):
    return None  # skip
```

#### `signal_statistics(signal)`

채널별 신호 품질 통계를 계산합니다.

```python
# signal: (samples, n_leads) ← transpose 후 전달
stats = signal_statistics(sig_reordered.T)
# 반환: {nan_ratio, amp_mean, amp_std, amp_skewness, amp_kurtosis}
# 각 값은 길이 12의 리스트
```

#### `beat_similarity(signal, sampling_rate)`

채널별 인접 박동 간 유사도를 계산합니다.

```python
bs = beat_similarity(sig_reordered.T, sampling_rate=500)
# 반환: {bs_corr, bs_dtw}
# bs_corr: Pearson 상관계수 평균 (1에 가까울수록 규칙적)
# bs_dtw:  정규화 DTW 거리 평균 (0에 가까울수록 규칙적)
```

#### `extract_beat_annotation(signal_lead2, sampling_rate)`

Lead II 신호에서 R-peak를 검출합니다.

```python
ba = extract_beat_annotation(sig_reordered[1], fs=500)
# 반환: {sample, symbol, subtype, chan, num, aux_note}
```

#### `extract_fiducial(signal_reordered, sampling_rate)`

Lead II 기준 DWT 파형 분할로 피듀셜 포인트와 19개 피처를 추출합니다.  
전기 축은 Lead I 과 Lead II 를 함께 사용합니다.

```python
# signal_reordered: (12, samples), 채널 순서 고정 필수
fp, ff = extract_fiducial(sig_reordered, fs=500)
# fp: {fsample: [...], fiducial: [...]}
# ff: {p_amp, q_amp, r_amp, s_amp, t_amp, p_dur, ...} float16 dict
```

---

### `verify_h5.py`

변환 결과를 검증하는 스크립트입니다.

**단일 파일 상세 확인:**

```bash
python verify_h5.py --file data/he110745030.h5
```

루트 속성, 메타데이터, 신호 통계, 어노테이션 내용을 모두 출력합니다.

**폴더 전체 일괄 검증:**

```bash
# 전체 검증
python verify_h5.py --dir /home/irteam/opendata1/h5/heedb/v4.0/data

# 샘플 100개만 무작위 검증
python verify_h5.py --dir /home/irteam/opendata1/h5/heedb/v4.0/data --sample 100
```

검증 항목:

| 항목 | 기준 |
|---|---|
| root attrs | `dataset_version`, `file_name` 존재 여부 |
| metadata attrs | `record_name`, `n_sig`, `fs`, `sig_len` 존재 여부 |
| sig_name | `['I','II','III','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR']` 일치 여부 |
| segment 존재 | `seg_len` 만큼의 세그먼트 키 존재 여부 |
| signal shape | `signal.shape[0] == 12` |

---

### `test_convert.py`

변환 파이프라인 전체를 단일 레코드로 테스트합니다.  
변환 → H5 저장 → 재로드 후 원본과 비교까지 수행합니다.

```bash
python test_convert.py
```

**스크립트 내 설정값:**

```python
INSTITUTION = "I0001"
BASE_DIR = f"/home/irteam/opendata1/raw/heedb/ECG/{INSTITUTION}"
META_PATH = os.path.join(BASE_DIR, "metadata", "metadata.csv")
WFDB_ROOT = os.path.join(BASE_DIR, "WFDB")
OUTPUT_H5 = "/home/irteam/tykim/convert_h5/convert_raw_to_h5/test_record.h5"
```

**검증 항목:**
- root attrs / metadata attrs 정상 저장 여부
- `sig_name` 일치 여부
- `signal` shape 및 `np.allclose` 원본 비교 (atol=0.01, float16 오차 허용)
- `beat_annotation` / `fiducial_point` / `fiducial_feature` 내용 확인
- CSV row 미리보기 (age 정규화, gender 인코딩 등)

---

## 신호 품질 지표 상세

`--compute_quality` (기본 활성화) 시 `heedb_table.csv` 에 아래 컬럼이 추가됩니다.

| 컬럼 | 설명 |
|---|---|
| `nan_ratio` | 채널별 NaN 샘플 비율 (0.0 ~ 1.0) |
| `amp_mean` | 채널별 진폭 평균 |
| `amp_std` | 채널별 진폭 표준편차 |
| `amp_skewness` | 채널별 왜도 |
| `amp_kurtosis` | 채널별 첨도 |
| `bs_corr` | 채널별 인접 박동 Pearson 상관계수 평균 |
| `bs_dtw` | 채널별 인접 박동 DTW 거리 평균 (정규화) |

모든 값은 길이 12의 Python 리스트를 문자열로 저장합니다.

`beat_similarity` 는 R-peak 검출 후 각 박동을 `sampling_rate × 2` 길이로 리샘플링하고 z-score 정규화한 뒤 인접 박동 간 지표를 계산합니다. R-peak 가 3개 이하이면 `NaN` 입니다.

---

## age 인코딩 규칙

```python
age = float(AgeAtAcquisition) / 365.25 / 100
```

`AgeAtAcquisition` 은 일(day) 단위로 저장된 나이입니다.  
365.25로 나누어 연(year) 단위로 변환한 뒤, 100으로 나누어 0~1 범위로 정규화합니다.  
변환 실패 시 `-1` 로 기록됩니다.

## gender 인코딩 규칙

| 원본 값 | 저장 값 |
|---|---|
| `"MALE"` | `1` |
| `"FEMALE"` | `-1` |
| 그 외 / 결측 | `0` |

I0001은 `SexDSC` 컬럼, I0006은 `Sex` 컬럼을 사용합니다.

---

## 자주 발생하는 문제

### H5 파일이 일부만 생성된 경우

재실행하면 이미 생성된 파일은 건너뛰고 나머지를 이어서 처리합니다.

```bash
python convert_to_h5_heedb.py
```

### `sig_name mismatch` 검증 오류

원본 WFDB 레코드의 채널 집합이 12 표준 유도와 다릅니다. 변환 시 이미 스킵 처리되므로 H5 파일에는 영향 없습니다.

### `beat_annotation` / `fiducial_feature` 가 H5에 없음

`--compute_beat` / `--compute_fiducial` 옵션 없이 변환된 파일입니다.  
해당 옵션을 추가하여 재변환하거나, 별도 후처리 스크립트로 추가 저장이 필요합니다.

### Ray 메모리 오류

`--batch_size` 를 줄이거나 `--num_cpus` 를 낮추어 재시도합니다.

```bash
python convert_to_h5_heedb.py --num_cpus 16 --batch_size 1000
```