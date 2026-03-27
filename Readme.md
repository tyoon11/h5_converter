# ECG H5 변환 파이프라인

공개 ECG 데이터셋을 내부 표준 H5 포맷으로 변환하는 파이프라인입니다.
HEEDB와 그 외 12개 공개 데이터셋을 동일한 H5 구조로 통일합니다.

---

## 프로젝트 구조

```
project/
├── heedb/                            # HEEDB 전용 파이프라인
│   ├── convert_to_h5_heedb.py        #   메인 변환 스크립트
│   ├── create_h5_structure_heedb.py  #   H5 구조 생성 함수 (public과 공유)
│   ├── utils_heedb.py                #   신호 처리 유틸리티 (public과 공유)
│   ├── test_convert.py               #   단일 레코드 변환 테스트
│   ├── verify_h5.py                  #   H5 검증 스크립트
│   └── README.md                     #   HEEDB 파이프라인 상세 문서
│
├── convert_to_h5_public.py           # 공개 데이터셋 변환 스크립트
├── test_convert_public.py            # 공개 데이터셋 단일 레코드 변환 테스트
├── verify_h5_public.py               # 공개 데이터셋 H5 검증 스크립트
├── append_signal_quality.py          # 신호 품질 계산 후 CSV에 추가
└── README.md                         # 이 파일
```

---

## 지원 데이터셋

### HEEDB

| 기관 코드 | 출처 | prefix | 규모 |
|-----------|------|--------|------|
| I0001 | MGH (Massachusetts General Hospital) | `he1` | ~10.6M ECG |
| I0006 | EUH (Emory University Hospital) | `he6` | ~1.0M ECG |

자세한 내용은 [`heedb/README.md`](heedb/README.md) 참조.

### 공개 데이터셋

| 키 | 데이터셋 | prefix | fs | 리드 수 |
|----|----------|--------|----|---------|
| `ptb_xl` | PTB-XL | `px` | 500 Hz | 12 |
| `ptb` | PTB | `pt` | 1000 Hz | 12 (15채널 중) |
| `sph` | SPH | `sp` | 500 Hz | 12 |
| `echonext` | EchoNext | `en` | 250 Hz | 12 |
| `zzu_pecg` | ZZU pECG | `zz` | 500 Hz | 12 |
| `code15` | CODE-15% | `c1` | 400 Hz | 8 → 12 * |
| `chapman` | Chapman | `ch` | 500 Hz | 12 |
| `cpsc2018` | CPSC 2018 | `cs` | 500 Hz | 12 |
| `cpsc_extra` | CPSC-Extra | `ce` | 500 Hz | 12 |
| `georgia` | Georgia | `ge` | 500 Hz | 12 |
| `ningbo` | Ningbo | `nb` | 500 Hz | 12 |
| `mimic` | MIMIC-IV-ECG | `mi` | 500 Hz | 12 |

\* CODE-15%는 I, II, V1–V6 8채널만 제공. 나머지 4채널(III, aVR, aVL, aVF)은 NaN으로 저장.

---

## H5 파일 구조

HEEDB와 공개 데이터셋 모두 동일한 구조를 사용합니다.

```
{file_name}.h5
├── attrs
│   ├── dataset_version           # str   "1.0"
│   ├── file_name                 # str   고유 파일명 식별자
│   ├── beat_ext_method           # str   "neurokit2" 또는 ""
│   └── fidu_extract_method       # str   "neurokit2-dwt" 또는 ""
└── ECG/
    ├── metadata/
    │   ├── attrs: record_name, n_sig(=12), fs, sig_len,
    │   │         base_time, base_date, dtype("fp16")
    │   └── datasets: sig_name, fmt, adc_gain, baseline,
    │                 units, adc_res, adc_zero
    └── segments/
        ├── attrs: seg_len
        └── 0/
            ├── signal             # float16, shape (12, timepoints)
            ├── beat_annotation/   # 선택 (--compute_beat)
            │   ├── sample         # int16, R-peak 샘플 인덱스
            │   ├── symbol         # UTF-8
            │   ├── subtype        # int16
            │   ├── chan           # int16
            │   ├── num            # int16
            │   └── aux_note       # UTF-8
            ├── fiducial_point/    # 선택 (--compute_fiducial)
            │   ├── fsample        # int16, 샘플 인덱스
            │   └── fiducial       # UTF-8, 파형 레이블
            └── fiducial_feature/  # 선택 (--compute_fiducial)
                └── attrs: p_amp, q_amp, r_amp, s_amp, t_amp,
                           p_dur, pr_seg, qrs_dur, st_seg, t_dur,
                           pr_int, qt_int, rr_int, tp_seg,
                           qtc_baz, qtc_frid, p_axis, r_axis, t_axis
```

### 리드 순서 (고정)

```python
TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']
#  index           [ 0,   1,    2,    3,   4,   5,   6,   7,   8,    9,   10,   11 ]
```

모든 데이터셋의 리드 순서를 위 순서로 통일합니다.
공개 데이터셋은 clinical_ts canonical 순서에서, HEEDB는 WFDB sig_name 기준으로 재정렬합니다.

```
canonical → TARGET 재정렬 인덱스: [0, 1, 2, 6, 7, 8, 9, 10, 11, 5, 4, 3]
```

---

## 환경 설정

```bash
pip install wfdb h5py ray tqdm numpy pandas neurokit2 dtw-python scipy
```

공개 데이터셋 변환에는 `clinical_ts` 라이브러리가 추가로 필요합니다.
프로젝트 루트의 `code/` 폴더에 설치하거나 `sys.path`에 추가하세요.

---

## 사용법

### 1. 변환 전 테스트

전체 변환 전에 파이프라인이 정상 동작하는지 소량의 레코드로 확인합니다.

```bash
# HEEDB — metadata.csv 500번째 행 1개 변환
python heedb/test_convert.py

# 공개 데이터셋 — 첫 3개 레코드 변환 (기본)
python test_convert_public.py \
    --dataset ptb_xl \
    --dataset_dir /data/ecg_datasets \
    --output_root /data/h5/public/v1.0

# beat + fiducial 포함 테스트
python test_convert_public.py \
    --dataset code15 \
    --dataset_dir /data/ecg_datasets \
    --output_root /data/h5/public/v1.0 \
    --n 2 --compute_beat --compute_fiducial
```

### 2. 전체 변환

```bash
# HEEDB — 신호만 저장 (기본, 가장 빠름)
python heedb/convert_to_h5_heedb.py --num_cpus 64

# HEEDB — beat + fiducial 포함
python heedb/convert_to_h5_heedb.py --num_cpus 64 --compute_beat --compute_fiducial

# 공개 데이터셋 — 단일
python convert_to_h5_public.py \
    --dataset_dir /data/ecg_datasets \
    --output_root /data/h5/public/v1.0 \
    --dataset ptb_xl \
    --num_cpus 32

# 공개 데이터셋 — 여러 개 (쉼표 구분)
python convert_to_h5_public.py \
    --dataset_dir /data/ecg_datasets \
    --output_root /data/h5/public/v1.0 \
    --dataset ptb_xl,sph,mimic \
    --num_cpus 32

# 공개 데이터셋 — 전체
python convert_to_h5_public.py \
    --dataset_dir /data/ecg_datasets \
    --output_root /data/h5/public/v1.0 \
    --dataset all \
    --num_cpus 32
```

변환은 **증분 방식**으로 동작합니다. 이미 존재하는 H5 파일은 건너뛰므로 중단 후 재실행해도 안전합니다.

### 3. 검증

```bash
# HEEDB — 단일 파일 상세 검증
python heedb/verify_h5.py --file /data/h5/heedb/v4.0/data/he1123450.h5

# HEEDB — 폴더 샘플 검증
python heedb/verify_h5.py --dir /data/h5/heedb/v4.0/data --sample 200

# 공개 데이터셋 — 단일 파일 상세 검증
python verify_h5_public.py --file /data/h5/public/v1.0/data/ptb_xl/px12345_0.h5

# 공개 데이터셋 — 특정 폴더 샘플 검증
python verify_h5_public.py --dir /data/h5/public/v1.0/data/ptb_xl --sample 200

# 공개 데이터셋 — output_root 전체 일괄 검증 (데이터셋별 결과 표 출력)
python verify_h5_public.py --output_root /data/h5/public/v1.0 --sample 200

# 공개 데이터셋 — 특정 데이터셋만 선택해 검증
python verify_h5_public.py \
    --output_root /data/h5/public/v1.0 \
    --dataset ptb_xl,mimic --sample 100
```

### 4. 신호 품질 계산 (변환 후 별도 실행)

신호 품질은 변환 속도 유지를 위해 변환 스크립트와 분리되어 있습니다.
변환 완료 후 아래 스크립트로 CSV에 품질 컬럼을 추가합니다.

```bash
# HEEDB
python append_signal_quality.py \
    --csv  /data/h5/heedb/v4.0/heedb_table.csv \
    --h5_root /data/h5/heedb/v4.0 \
    --num_cpus 64 \
    --backup

# 공개 데이터셋 전체
python append_signal_quality.py \
    --csv  /data/h5/public/v1.0/public_ecg_table.csv \
    --h5_root /data/h5/public/v1.0 \
    --num_cpus 32

# 특정 데이터셋만
python append_signal_quality.py \
    --csv  /data/h5/public/v1.0/public_ecg_table.csv \
    --h5_root /data/h5/public/v1.0 \
    --dataset ptb_xl,sph \
    --num_cpus 32

# DTW 생략 (속도 우선, bs_dtw = NaN)
python append_signal_quality.py \
    --csv  /data/h5/public/v1.0/public_ecg_table.csv \
    --h5_root /data/h5/public/v1.0 \
    --no_dtw --num_cpus 64
```

---

## 출력 파일

### HEEDB

```
/home/irteam/opendata1/h5/heedb/v4.0/
├── data/
│   ├── he1{pid}{rid}.h5       # I0001 H5 파일
│   └── he6{pid}{rid}.h5       # I0006 H5 파일
├── heedb_table.csv            # 학습용 메타 테이블
├── file_name.csv              # 원본 ↔ H5 파일명 매핑
├── combined_metadata.csv      # 양 기관 원본 메타데이터 통합
└── conversion.log
```

### 공개 데이터셋

```
/data/h5/public/v1.0/
├── data/
│   ├── ptb_xl/     px{pid}_{rid}.h5
│   ├── ptb/        pt{pid}_{rid}.h5
│   ├── sph/        sp{pid}_{rid}.h5
│   └── ...
├── public_ecg_table.csv       # 전체 통합 테이블
├── ptb_xl_table.csv           # 데이터셋별 개별 테이블
├── ptb_table.csv
├── ...
└── conversion_public.log
```

### CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| `filepath` | H5 파일 상대 경로 (`data/{dataset}/{file_name}.h5`) |
| `dataset` | 데이터셋 키 (공개 데이터셋만. HEEDB는 없음) |
| `pid` | 환자 ID |
| `rid` | 원본 메타데이터 행 인덱스 |
| `sid` | 세그먼트 인덱스 (현재 항상 `0`) |
| `oid` | 고유 관측 ID |
| `age` | 나이 (years / 100, 0~1 범위) |
| `gender` | 1=남성, -1=여성, 0=미상 |
| `height` | cm (미제공 시 NaN) |
| `weight` | kg (미제공 시 NaN) |
| `fs` | 샘플링 주파수 (Hz) |
| `channel_name` | TARGET_SIG_NAME 리스트 문자열 |
| `nan_ratio` | *(품질 계산 후)* per-lead NaN 비율 (12개 리스트) |
| `amp_mean` | *(품질 계산 후)* per-lead 진폭 평균 |
| `amp_std` | *(품질 계산 후)* per-lead 진폭 표준편차 |
| `amp_skewness` | *(품질 계산 후)* per-lead 왜도 |
| `amp_kurtosis` | *(품질 계산 후)* per-lead 첨도 |
| `bs_corr` | *(품질 계산 후)* per-lead beat-to-beat 상관계수 |
| `bs_dtw` | *(품질 계산 후)* per-lead beat-to-beat DTW 거리 |

---

## CLI 옵션 요약

### `convert_to_h5_public.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset_dir` | 필수 | 원본 데이터셋 루트 경로 |
| `--output_root` | 필수 | H5 출력 루트 경로 |
| `--dataset` | `all` | 변환할 데이터셋 (all / 쉼표 구분 / 단일) |
| `--num_cpus` | `32` | Ray 병렬 CPU 수 |
| `--batch_size` | `2000` | 배치 크기 |
| `--compute_beat` | OFF | beat_annotation 생성 |
| `--compute_fiducial` | OFF | fiducial_point / feature 생성 |
| `--compute_quality` | OFF | 신호 품질 계산 (권장하지 않음) |

### `test_convert_public.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | 필수 | 테스트할 데이터셋 키 (예: `ptb_xl`) |
| `--dataset_dir` | 필수 | 원본 데이터셋 루트 경로 |
| `--output_root` | 필수 | H5 출력 루트 경로 |
| `--n` | `3` | 테스트할 레코드 수 |
| `--compute_beat` | OFF | beat_annotation 추출 포함 |
| `--compute_fiducial` | OFF | fiducial_point / feature 추출 포함 |

### `verify_h5_public.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--file` | — | 단일 H5 파일 상세 검증 |
| `--dir` | — | 특정 폴더 일괄 검증 |
| `--output_root` | — | output_root/data/ 전체 데이터셋 일괄 검증 |
| `--dataset` | 전체 | 검증할 데이터셋 (쉼표 구분, `--output_root` 모드) |
| `--sample` | 전체 | 폴더당 무작위 샘플링 파일 수 |
| `--allow_nan_leads` | OFF | 전체 NaN 리드를 오류로 취급하지 않음 |

### `append_signal_quality.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--csv` | 필수 | 대상 CSV 경로 |
| `--h5_root` | 필수 | H5 루트 경로 |
| `--dataset` | 전체 | 특정 데이터셋만 처리 (쉼표 구분) |
| `--overwrite` | OFF | 이미 계산된 행 재계산 |
| `--no_dtw` | OFF | DTW 생략 (`bs_dtw = NaN`) |
| `--num_cpus` | `32` | Ray 병렬 CPU 수 |
| `--batch_size` | `1000` | Ray 배치 크기 |
| `--save_interval` | `10` | N 배치마다 중간 저장 (0이면 최종 1회만) |
| `--backup` | OFF | 처리 전 원본 CSV를 `.backup.csv` 로 백업 |

---

## 권장 실행 순서

```
1. 변환 테스트      python heedb/test_convert.py
                    python test_convert_public.py \
                        --dataset ptb_xl \
                        --dataset_dir /data/raw \
                        --output_root /data/h5/public/v1.0

2. 전체 변환        python heedb/convert_to_h5_heedb.py --num_cpus 64
                    python convert_to_h5_public.py \
                        --dataset_dir /data/raw \
                        --output_root /data/h5/public/v1.0 \
                        --dataset all --num_cpus 32

3. 검증             python heedb/verify_h5.py \
                        --dir /data/h5/heedb/v4.0/data --sample 500
                    python verify_h5_public.py \
                        --output_root /data/h5/public/v1.0 --sample 200

4. 신호 품질 계산   python append_signal_quality.py \
                        --csv /data/h5/heedb/v4.0/heedb_table.csv \
                        --h5_root /data/h5/heedb/v4.0 --num_cpus 64
                    python append_signal_quality.py \
                        --csv /data/h5/public/v1.0/public_ecg_table.csv \
                        --h5_root /data/h5/public/v1.0 --num_cpus 32
```

---

## 주의 사항

- **신호 품질 계산은 변환과 분리**합니다. 변환 스크립트의 `--compute_quality` 옵션은 속도 저하가 크므로 `append_signal_quality.py` 사용을 권장합니다.
- `append_signal_quality.py` 는 **재개(Resume)를 지원**합니다. `nan_ratio` 컬럼이 비어있는 행만 계산하므로 중단 후 재실행해도 안전합니다.
- **CODE-15%** 는 III, aVR, aVL, aVF 4개 리드가 NaN입니다. `verify_h5_public.py` 는 해당 폴더명을 자동 감지해 NaN 리드를 허용합니다. 모델 학습 시에는 별도 마스킹 처리가 필요합니다.
- 공개 데이터셋 변환은 `clinical_ts` 라이브러리에 의존합니다. `code/` 폴더에 설치되어 있어야 합니다.