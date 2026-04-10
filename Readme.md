# ECG H5 Converter

HEEDB 및 공개 ECG 데이터셋을 동일한 표준 H5 포맷으로 변환하는 통합 파이프라인입니다.

---

## 프로젝트 구조

```
convert_raw_to_h5/
├── convert_to_h5.py              # 통합 변환 (HEEDB + 공개 데이터셋, 2~12리드 지원)
├── append_fiducial.py            # beat_annotation / fiducial 후처리 (멀티 세그먼트)
├── append_signal_quality.py      # 신호 품질 지표 후처리 (bs_corr, bs_dtw)
├── append_labels.py              # 라벨 맵핑 → 데이터셋별 라벨 CSV/JSON 생성
├── verify_h5.py                  # H5 검증 (데이터셋별 상세 + 일괄, 가변 리드)
├── convert_old_h5_to_new.py      # 구버전 H5 → 신규 포맷 마이그레이션
├── mimic_preprocessing.py        # MIMIC-IV-ECG 전용 전처리
├── run_convert.sh                # 전체 변환 파이프라인
├── run_test_verify.sh            # 테스트 + 검증 일괄 실행
├── test_convert.py               # 데이터셋별 1개 변환 테스트 + CSV 컬럼 검증
└── utils/
    ├── h5_structure.py           # H5 구조 생성 (create_h5_structure)
    └── signal_processing.py      # 신호 처리 (reorder, statistics, beat, fiducial)
```

---

## 데이터셋 현황 (전수 검사 결과)

### 12리드 데이터셋

| 데이터셋 | 그룹 | prefix | H5 파일수 | fs(Hz) | raw 길이 | seg_len | 리드 | age | gender |
|---|---|---|---:|---|---|---|---:|---|---|
| **heedb_i0001** (MGH) | heedb | `he1` | 10,269,531 | 500/250 | 10s | 1 | 12 | 62.8±18.3 | M:5.9M/F:5.3M |
| **heedb_i0006** (EUH) | heedb | `he6` | 969,699 | 500/250 | 10s | 1 | 12 | — | — |
| **chapman** | physionet | `psh` | 10,232 | 500 | 10s | 1 | 12 | 60.1±17.0 | M:5,726/F:4,506 |
| **cpsc2018** | physionet | `pcp` | 6,849 | 500 | 10~40s | 1~4 | 12 | 60.2±19.0 | M:3,681/F:3,168 |
| **cpsc_extra** | physionet | `pce` | 3,433 | 500 | 10~40s | 1~4 | 12 | 63.7±15.4 | M:1,835/F:1,598 |
| **georgia** | physionet | `pge` | 10,283 | 500 | 10s | 1 | 12 | 60.5±15.4 | M:5,516/F:4,767 |
| **ningbo** | physionet | `pnb` | 33,425 | 500 | 10s | 1 | 12 | 59.5±18.1 | M:18,906/F:14,497 |
| **ptb** | physionet | `ppt` | 516 | 1000 | 30~120s | 3~12 | 12 | 56.3±14.0 | M:377/F:138 |
| **ptbxl** | physionet | `ppx` | 21,836 | 500 | 10s | 1 | 12 | 59.5±16.8 | M:11,379/F:10,457 |
| **stpetersburg** | physionet | `pin` | 74 | 257 | 1800s | 180 | 12 | 56.0±13.9 | M:40/F:34 |
| **zzu_pecg** (소아) | zzu | `zzu` | 12,327 | 500 | 10~60s | 1~6 | 12 | 9.1±4.1 | M:7,130/F:5,197 |
| **code15** | code15 | `cod` | 345,779 | 400 | 10s | 1 | 12 | 53.2±19.7 | M:139k/F:207k |
| **mimic4** | mimic4 | `m4p` | 799,929 | 500 | 10s | 1 | 12 | — | — |

### 2리드 데이터셋

| 데이터셋 | 그룹 | prefix | H5 파일수 | fs(Hz) | raw 길이 | seg_len | 리드 | 라벨 |
|---|---|---|---:|---|---|---|---:|---|
| **cpsc2021** | cpsc2021 | `c21` | 1,425 | 200 | 10~24,660s | 1~2,466 | 2 (I, II) | AF 3종 |

### 합계

- **전체 H5 파일: 12,495,338개**
- **전체 세그먼트: ~12.7M개** (대부분 seg_len=1, 장시간 레코드는 수백)

---

## 라벨 현황

각 데이터셋의 H5 폴더에 `{dataset}_labels.csv` + `{dataset}_labels.json`으로 저장됩니다.

| 데이터셋 | 라벨 CSV 경로 | 라벨 수 | 맵핑률 | 라벨 소스 |
|---|---|---:|---|---|
| **heedb** | `heedb/v4.0/heedb_labels.csv` | 149 | 89.7% | 12SL GE algorithm (diagnoses.csv) |
| **physionet** | `physionet/v2.0/physionet_labels.csv` | 31 | 92.9% | SNOMED codes (Challenge 2021) |
| **zzu** | `ZZU-pECG/v2.0/zzu_labels.csv` | 36 | 92.3% | AHA/CHN codes |
| **cpsc2021** | `cpsc2021/v2.0/cpsc2021_labels.csv` | 3 | 100% | .hea comments |
| **code15** | `code15/v2.0/code15_labels.csv` | 6 | 100% | 원본 CSV |
| **mimic4** | — | 0 | — | 별도 파이프라인 필요 |

### 라벨 CSV 구조

```csv
filepath,dataset,pid,rid,oid,is_AF,is_NSR,...
data/pshJS000010.h5,chapman,JS00001,0,pshJS0000100,False,True,...
```

- **key 컬럼**: `filepath`, `pid`, `rid`, `oid` (+ physionet/zzu/cpsc2021은 `dataset`)
- **라벨 컬럼**: binary (`True`/`False`)
- table CSV (`ecg_table.csv`)에는 라벨 미포함 — 조인하여 사용

### 주요 라벨 예시

| heedb (149) | physionet (31) | zzu (36) | cpsc2021 (3) | code15 (6) |
|---|---|---|---|---|
| ATRIAL_FIBRILLATION | is_AF | is_STach | is_nonAF | 1dAVb |
| NORMAL_SINUS_RHYTHM | is_NSR | is_Normal | is_persistentAF | RBBB |
| SINUS_BRADYCARDIA | is_RBBB | is_RVH | is_paroxysmalAF | LBBB |
| LEFT_BUNDLE_BRANCH_BLOCK | is_LBBB | is_IRBBB | | is_SB |
| INFERIOR_INFARCT | is_STach | is_QAb | | is_STach |
| ACUTE_MI_STEMI | is_LQT | is_LQT | | is_AF |

---

## H5 파일 구조

### 12리드 템플릿

```
pshJS000010.h5
├── @dataset_version    = "1.0"
├── @file_name          = "pshJS000010"
├── @beat_ext_method    = "neurokit2"
├── @fidu_extract_method = "neurokit2-dwt"
└── ECG/
    ├── metadata/
    │   ├── @record_name  = "JS00001"
    │   ├── @n_sig        = 12
    │   ├── @fs           = 500
    │   ├── @sig_len      = 5000
    │   ├── @base_time    = ""
    │   ├── @base_date    = ""
    │   ├── @dtype         = "fp16"
    │   ├── sig_name       (12,)   ['I','II','III','V1',...,'aVR']
    │   ├── fmt            (12,)   ['16','16',...]
    │   ├── adc_gain       (12,)   float32
    │   ├── baseline       (12,)   int32
    │   ├── units          (12,)   ['mV','mV',...]
    │   ├── adc_res        (12,)   int16
    │   └── adc_zero       (12,)   int16
    └── segments/
        ├── @seg_len = 1
        └── 0/
            ├── signal              (12, 5000)   float16
            ├── beat_annotation/
            │   ├── sample          (N,)         int16    # R-peak 위치
            │   ├── symbol          (N,)         str
            │   ├── subtype         (N,)         int16
            │   ├── chan            (N,)         int16
            │   ├── num             (N,)         int16
            │   └── aux_note        (N,)         str
            ├── fiducial_point/
            │   ├── fsample         (M,)         int16
            │   └── fiducial        (M,)         str
            └── fiducial_feature/
                ├── @p_amp, @q_amp, @r_amp, @s_amp, @t_amp        float16
                ├── @p_dur, @pr_seg, @qrs_dur, @st_seg, @t_dur    float16
                ├── @pr_int, @qt_int, @rr_int, @tp_seg             float16
                ├── @qtc_baz, @qtc_frid                             float16
                └── @p_axis, @r_axis, @t_axis                      float16
```

### 2리드 템플릿 (cpsc2021)

```
c21data_0_10.h5
├── @file_name          = "c21data_0_10"
└── ECG/
    ├── metadata/
    │   ├── @n_sig        = 2
    │   ├── @fs           = 200
    │   └── sig_name       (2,)   ['I','II']
    └── segments/
        ├── @seg_len = 104          # 원본 ~17분 → 104개 × 10초
        ├── 0/
        │   └── signal   (2, 2000)  float16
        ├── 1/
        │   └── signal   (2, 2000)  float16
        └── .../
```

### 멀티 세그먼트 예시 (ptb, seg_len=11)

```
pptS00010.h5
└── ECG/segments/
    ├── @seg_len = 11               # ~110초 녹화 → 11개 × 10초
    ├── 0/signal  (12, 10000)       # fs=1000 × 10s
    ├── 1/signal  (12, 10000)
    └── .../10/signal (12, 10000)
```

### 세그먼트 분할 규칙

- 신호 ≥ 10초 → `fs * 10` 샘플 단위로 분할 (`seg_len = N`)
- 끝의 10초 미만 잔여는 버림
- 신호 < 10초 → 1개 세그먼트로 그대로 저장
- `beat_annotation` / `fiducial_point` / `fiducial_feature`는 세그먼트별 독립 계산
- `append_signal_quality.py`는 전 세그먼트를 concat 후 통계 계산

### 리드 순서 (12리드 표준)

```python
TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']
```

2리드 데이터셋은 원본 리드 순서 그대로 저장 (`target_sig_name=None`).

---

## 출력 디렉토리 구조

```
/home/irteam/ddn-opendata1/h5/
├── heedb/v4.0/
│   ├── data/                     # he1*.h5, he6*.h5
│   ├── heedb_table.csv           # 메타 테이블 (라벨 미포함)
│   ├── heedb_labels.csv          # 라벨 테이블 (key + 149 binary 컬럼)
│   ├── heedb_labels.json         # 라벨 정의
│   ├── file_name.csv
│   └── combined_metadata.csv
├── physionet/v2.0/
│   ├── data/                     # psh*.h5, pcp*.h5, ...
│   ├── ecg_table.csv             # 통합 메타 테이블
│   ├── {dataset}_table.csv       # 데이터셋별 테이블
│   ├── physionet_labels.csv      # 라벨 (31개)
│   ├── physionet_labels.json
│   └── file_name.csv
├── ZZU-pECG/v2.0/
│   ├── data/                     # zzu*.h5
│   ├── ecg_table.csv
│   ├── zzu_labels.csv            # 라벨 (36개)
│   └── zzu_labels.json
├── cpsc2021/v2.0/
│   ├── data/                     # c21*.h5
│   ├── ecg_table.csv
│   ├── cpsc2021_labels.csv       # 라벨 (3개: AF 분류)
│   └── cpsc2021_labels.json
├── code15/v2.0/
│   ├── data/                     # cod*.h5
│   ├── code15_table.csv
│   ├── code15_labels.csv         # 라벨 (6개)
│   └── code15_labels.json
└── mimic4/v2.0/
    ├── data/                     # m4p*.h5
    └── mimic4_table.csv
```

### table CSV 컬럼 (라벨 미포함)

| 컬럼 | 설명 |
|------|------|
| `filepath` | H5 상대 경로 (`data/xxx.h5`) |
| `dataset` | 데이터셋 키 (예: `ptbxl`, `heedb_i0001`) |
| `pid` | 환자 ID |
| `rid` | 레코드 인덱스 |
| `sid` | 세그먼트 ID (항상 `0`) |
| `oid` | 고유 ID (`{prefix}{pid}{rid}{sid}`) |
| `age` | 나이 (years/100) |
| `gender` | 1=남, -1=여, 0=미상 |
| `height` / `weight` | 키/몸무게 (NaN) |
| `fs` | 샘플링 주파수 (Hz) |
| `channel_name` | 채널 순서 |
| `nan_ratio` ~ `amp_kurtosis` | per-lead 신호 통계 |
| `bs_corr` / `bs_dtw` | beat-to-beat 유사도 (별도 계산) |

---

## 사용법

### 1. 변환

```bash
# 12리드 데이터셋
python convert_to_h5.py --group physionet \
    --physionet_root /data/raw/physionet.org/files \
    --output_root /data/h5/physionet/v2.0 --num_cpus 64

# 2리드 데이터셋 (cpsc2021)
python convert_to_h5.py --group cpsc2021 \
    --cpsc2021_root /data/raw/physionet.org/files/cpsc2021 \
    --output_root /data/h5/cpsc2021/v2.0 --num_cpus 64

# 전체 (heedb + physionet + cpsc2021 + zzu)
python convert_to_h5.py --group all --output_root /data/h5/all/v1.0
```

### 2. 후처리

```bash
# fiducial (beat_annotation + fiducial_point/feature)
python append_fiducial.py \
    --csv /data/h5/physionet/v2.0/ecg_table.csv \
    --h5_root /data/h5/physionet/v2.0 --num_cpus 64

# 신호 품질 (bs_corr, bs_dtw)
python append_signal_quality.py \
    --csv /data/h5/physionet/v2.0/ecg_table.csv \
    --h5_root /data/h5/physionet/v2.0 --num_cpus 64
```

### 3. 라벨 맵핑

```bash
# 전체 데이터셋 라벨 생성
python append_labels.py --all

# 특정 데이터셋만
python append_labels.py --dataset physionet

# 라벨 정의 JSON만 (CSV 생성 없이)
python append_labels.py --dataset heedb --dry_run

# table CSV에서 라벨 컬럼 제거 (라벨 CSV 분리 후)
python append_labels.py --all --clean_table
```

### 4. 검증

```bash
python verify_h5.py --output_root /data/h5/physionet/v2.0 --sample 200
python verify_h5.py --file /data/h5/physionet/v2.0/data/pshJS000010.h5
```

### 5. 테스트

```bash
bash run_test_verify.sh all    # test_convert + verify_h5 일괄
bash run_convert.sh physionet  # 변환 + fiducial + signal_quality 일괄
```

---

## 권장 실행 순서

```
1. 변환          python convert_to_h5.py --group ... --output_root ...
2. fiducial      python append_fiducial.py --csv ecg_table.csv --h5_root ...
3. signal quality python append_signal_quality.py --csv ecg_table.csv --h5_root ...
4. 라벨 맵핑      python append_labels.py --all
5. 검증          python verify_h5.py --output_root ...
```

---

## 주의 사항

- **증분 변환**: 기존 H5 파일은 건너뜁니다. 재변환은 파일 삭제 후 재실행.
- **2리드 지원**: `convert_to_h5.py`에서 config의 `target_sig_name=None`으로 설정 시 원본 리드 그대로 저장.
- **라벨 분리**: table CSV에는 라벨 미포함. `{dataset}_labels.csv`를 `filepath`/`oid`로 조인하여 사용.
- **MIMIC-IV-ECG**: 라벨 맵핑 미지원 (별도 `mimic_preprocessing.py` 파이프라인 필요).
