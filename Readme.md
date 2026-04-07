# ECG H5 Converter

HEEDB 및 공개 ECG 데이터셋을 동일한 표준 H5 포맷으로 변환하는 통합 파이프라인입니다.

---

## 프로젝트 구조

```
ecg_h5_converter/
├── convert_to_h5.py              # 통합 변환 스크립트 (HEEDB + 공개 데이터셋)
├── append_fiducial.py            # fiducial point/feature 후처리 추가
├── append_signal_quality.py      # beat similarity (bs_corr/bs_dtw) 후처리 추가
├── verify_h5.py                  # H5 파일 검증
├── test_convert.py               # 단일 레코드 변환 테스트
├── convert_old_h5_to_new.py      # 구버전 H5 → 신규 포맷 마이그레이션
├── mimic_preprocessing.py        # MIMIC-IV-ECG 전용 전처리
├── run_convert.sh                # 전체 파이프라인 실행 스크립트
├── utils/                        # 공유 유틸리티
│   ├── __init__.py
│   ├── h5_structure.py           #   H5 구조 생성 (create_h5_structure)
│   └── signal_processing.py      #   신호 처리 (reorder, statistics, beat, fiducial)
├── code/                         # ecg-fm-benchmarking/code/ 복사본
│   └── clinical_ts/              #   (memmap 방식 전처리용, H5 변환에는 불필요)
└── README.md
```

---

## 사전 준비

### 1. 환경 설치

```bash
pip install wfdb h5py ray tqdm neurokit2 dtw-python scipy pandas numpy
```

### 2. code/ 폴더 (선택)

`code/clinical_ts/`는 memmap 방식 전처리(`preprocess_ecg_dataset.ipynb`)에만 사용됩니다.
H5 변환 파이프라인 자체는 `code/`가 없어도 동작합니다.

```bash
git clone https://github.com/AI4HealthUOL/ecg-fm-benchmarking.git
cp -r ecg-fm-benchmarking/code ./
```

### 3. Label mappings (PhysioNet CinC 2021 데이터셋)

CPSC2018, CPSC-Extra, Georgia, Ningbo 데이터셋은 각 폴더에 `Label mappings 2021.xlsx`가 필요합니다.

---

## 지원 데이터셋

### HEEDB

| 데이터셋 | 기관 | prefix | 규모 |
|----------|------|--------|------|
| `heedb_i0001` | MGH (Massachusetts General Hospital) | `he1` | ~10.6M ECG |
| `heedb_i0006` | EUH (Emory University Hospital) | `he6` | ~1.0M ECG |

### 공개 데이터셋

| 키 | 데이터셋 | prefix | fs | 비고 |
|----|----------|--------|----|------|
| `chapman` | Chapman-Shaoxing | `psh` | 500 Hz | |
| `cpsc2018` | CPSC 2018 | `pcp` | 500 Hz | CinC 2021 |
| `cpsc_extra` | CPSC-Extra | `pce` | 500 Hz | CinC 2021 |
| `georgia` | Georgia | `pge` | 500 Hz | CinC 2021 |
| `ningbo` | Ningbo | `pnb` | 500 Hz | CinC 2021 |
| `ptb` | PTB | `ppt` | 1000 Hz | |
| `ptbxl` | PTB-XL | `ppx` | 500 Hz | |
| `stpetersburg` | St. Petersburg INCART | `pin` | 257 Hz | |
| `zzu_pecg` | ZZU pECG | `zzu` | 500 Hz | 소아 ECG |

---

## 사용법

### 1. 변환 테스트 (단일 레코드)

```bash
python test_convert.py \
    --dataset ptbxl \
    --physionet_root /data/raw/physionet.org/files \
    --output_root /tmp/test_h5
```

### 2. 전체 변환

```bash
# HEEDB 전체
python convert_to_h5.py --group heedb \
    --heedb_root /data/raw/heedb/ECG \
    --output_root /data/h5/heedb/v4.0 \
    --num_cpus 64

# PhysioNet 전체
python convert_to_h5.py --group physionet \
    --physionet_root /data/raw/physionet.org/files \
    --output_root /data/h5/physionet/v2.0 \
    --num_cpus 64

# ZZU
python convert_to_h5.py --group zzu \
    --zzu_root /data/raw/ZZU-pECG \
    --output_root /data/h5/zzu/v2.0

# 전체 (HEEDB + PhysioNet + ZZU)
python convert_to_h5.py --group all \
    --heedb_root /data/raw/heedb/ECG \
    --physionet_root /data/raw/physionet.org/files \
    --zzu_root /data/raw/ZZU-pECG \
    --output_root /data/h5/all/v1.0

# 특정 데이터셋만
python convert_to_h5.py --dataset georgia,ptbxl,heedb_i0001 \
    --physionet_root /data/raw/physionet.org/files \
    --heedb_root /data/raw/heedb/ECG \
    --output_root /data/h5/mixed/v1.0
```

변환은 **증분 방식**으로 동작합니다. 기존 H5 파일은 건너뛰므로 중단 후 재실행이 안전합니다.

### 3. 후처리

```bash
# fiducial point/feature 추가
python append_fiducial.py \
    --csv /data/h5/physionet/v2.0/ecg_table.csv \
    --h5_root /data/h5/physionet/v2.0 \
    --num_cpus 64

# beat similarity 추가 (bs_corr, bs_dtw)
python append_signal_quality.py \
    --csv /data/h5/physionet/v2.0/ecg_table.csv \
    --h5_root /data/h5/physionet/v2.0 \
    --num_cpus 64

# DTW 생략 (속도 우선)
python append_signal_quality.py \
    --csv /data/h5/physionet/v2.0/ecg_table.csv \
    --h5_root /data/h5/physionet/v2.0 \
    --no_dtw --num_cpus 64
```

### 4. 검증

```bash
python verify_h5.py --output_root /data/h5/physionet/v2.0 --sample 200
python verify_h5.py --file /data/h5/physionet/v2.0/data/psh12340.h5
```

### 5. 일괄 실행

`run_convert.sh`에서 경로를 수정한 후 실행:

```bash
bash run_convert.sh
```

---

## H5 파일 구조

모든 데이터셋이 동일한 구조를 사용합니다.

```
{file_name}.h5
├── attrs
│   ├── dataset_version           # str   "1.0"
│   ├── file_name                 # str   고유 식별자
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
            ├── beat_annotation/   # 선택
            │   ├── sample, symbol, subtype, chan, num, aux_note
            ├── fiducial_point/    # 선택
            │   ├── fsample, fiducial
            └── fiducial_feature/  # 선택
                └── attrs: p_amp, q_amp, r_amp, s_amp, t_amp, ...
```

### 리드 순서 (고정)

```python
TARGET_SIG_NAME = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']
```

---

## 출력 구조

```
output_root/
├── data/
│   ├── he1{pid}{rid}.h5       # HEEDB I0001
│   ├── he6{pid}{rid}.h5       # HEEDB I0006
│   ├── psh{pid}{rid}.h5       # Chapman
│   ├── pcp{pid}{rid}.h5       # CPSC2018
│   └── ...
├── ecg_table.csv              # 전체 통합 테이블
├── {dataset}_table.csv        # 데이터셋별 개별 테이블
├── file_name.csv              # 원본 ↔ H5 파일명 매핑
├── combined_metadata.csv      # HEEDB 양 기관 메타데이터 통합 (HEEDB만)
└── conversion.log
```

### CSV 컬럼

| 컬럼 | 시점 | 설명 |
|------|------|------|
| `filepath` | 변환 시 | H5 상대 경로 (`data/xxx.h5`) |
| `dataset` | 변환 시 | 데이터셋 키 |
| `pid` | 변환 시 | 환자 ID |
| `rid` | 변환 시 | 레코드 인덱스 |
| `sid` | 변환 시 | 세그먼트 (항상 `0`) |
| `oid` | 변환 시 | 고유 관측 ID |
| `age` | 변환 시 | 나이 (years/100, 0~1.5) |
| `gender` | 변환 시 | 1=남, -1=여, 0=미상 |
| `fs` | 변환 시 | 샘플링 주파수 (Hz) |
| `nan_ratio` ~ `amp_kurtosis` | 변환 시 | per-lead 신호 통계 |
| `bs_corr` | **별도 계산** | per-lead beat-to-beat 상관계수 |
| `bs_dtw` | **별도 계산** | per-lead beat-to-beat DTW 거리 |

---

## CLI 옵션

### `convert_to_h5.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--group` | — | `heedb` / `physionet` / `zzu` / `all` |
| `--dataset` | — | 쉼표 구분 데이터셋명 |
| `--heedb_root` | — | HEEDB ECG 루트 (I0001, I0006 상위) |
| `--physionet_root` | — | PhysioNet 데이터 루트 |
| `--zzu_root` | — | ZZU-pECG 루트 |
| `--output_root` | 필수 | H5 출력 루트 |
| `--num_cpus` | 64 | Ray CPU 수 |
| `--batch_size` | 2000 | 배치 크기 |
| `--compute_beat` | OFF | beat_annotation 생성 |
| `--compute_fiducial` | OFF | fiducial 생성 |

### `append_signal_quality.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--csv` | 필수 | 대상 CSV 경로 |
| `--h5_root` | 필수 | H5 루트 |
| `--no_dtw` | OFF | DTW 생략 |
| `--overwrite` | OFF | 기존 값 재계산 |
| `--backup` | OFF | 원본 CSV 백업 |
| `--num_cpus` | 32 | Ray CPU 수 |
| `--save_interval` | 10 | N 배치마다 중간 저장 |

---

## 권장 실행 순서

```
1. 테스트         python test_convert.py --dataset ptbxl ...

2. 전체 변환      python convert_to_h5.py --group all ...
                   → ecg_table.csv 자동 생성

3. fiducial 추가  python append_fiducial.py --csv ecg_table.csv ...

4. 검증           python verify_h5.py --output_root ... --sample 500

5. beat_similarity python append_signal_quality.py --csv ecg_table.csv ...
                   → bs_corr, bs_dtw 추가
```

---

## 주의 사항

- **증분 변환**: 이미 존재하는 H5는 자동으로 건너뜁니다. 재변환하려면 파일을 삭제 후 재실행하세요.
- **`code/` 폴더**: `clinical_ts` 패키지 포함. H5 변환에는 불필요하며, memmap 전처리(`preprocess_ecg_dataset.ipynb`)에만 사용됩니다.
- **beat_similarity**: 계산 비용이 크므로 변환과 분리하여 `append_signal_quality.py`로 실행합니다. Resume을 지원합니다.
- **MIMIC-IV-ECG**: 변환 전 `python mimic_preprocessing.py`를 먼저 실행해야 합니다.