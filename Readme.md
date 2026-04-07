# ECG H5 Converter

HEEDB 및 공개 ECG 데이터셋을 동일한 표준 H5 포맷으로 변환하는 통합 파이프라인입니다.

---

## 프로젝트 구조

```
h5_converter/
├── convert_to_h5.py              # 통합 변환 스크립트 (HEEDB + 공개 데이터셋)
├── append_fiducial.py            # fiducial point/feature 후처리 추가
├── append_signal_quality.py      # 신호 품질 지표(bs_corr/bs_dtw) 후처리 추가
├── verify_h5.py                  # H5 파일 검증 (데이터셋별 1개 상세 + 일괄)
├── convert_old_h5_to_new.py      # 구버전 H5 → 신규 포맷 마이그레이션
├── mimic_preprocessing.py        # MIMIC-IV-ECG 전용 전처리
├── run_convert.sh                # 전체 변환 파이프라인 일괄 실행 스크립트
├── run_test_verify.sh            # 테스트 + 검증 일괄 실행 스크립트
├── test_convert.py               # 전체 데이터셋 변환 테스트 (데이터셋별 1개) + CSV 컬럼 검증
├── utils/
│   ├── __init__.py
│   ├── h5_structure.py           # H5 구조 생성 (create_h5_structure)
│   └── signal_processing.py      # 신호 처리 (reorder, statistics, beat, fiducial)
└── _legacy/                      # 구버전 파일 (참조용 보관)
    ├── convert_to_h5_public.py   # → convert_to_h5.py --group physionet/zzu 로 대체
    ├── run_convert_public.sh     # → run_convert.sh 로 대체
    ├── verify_h5_public.py       # → verify_h5.py 로 대체
    └── heedb/                    # → utils/ 로 이전됨
```

---

## 사전 준비

```bash
pip install wfdb h5py ray tqdm neurokit2 dtw-python scipy pandas numpy
```

---

## 지원 데이터셋

### HEEDB

| 데이터셋 | 기관 | prefix | 규모 |
|----------|------|--------|------|
| `heedb_i0001` | MGH (Massachusetts General Hospital) | `he1` | ~10.6M ECG |
| `heedb_i0006` | EUH (Emory University Hospital) | `he6` | ~1.0M ECG |

### 공개 데이터셋

| 키 | 데이터셋 | prefix | fs |
|----|----------|--------|----|
| `chapman` | Chapman-Shaoxing | `psh` | 500 Hz |
| `cpsc2018` | CPSC 2018 | `pcp` | 500 Hz |
| `cpsc_extra` | CPSC-Extra | `pce` | 500 Hz |
| `georgia` | Georgia | `pge` | 500 Hz |
| `ningbo` | Ningbo | `pnb` | 500 Hz |
| `ptb` | PTB | `ppt` | 1000 Hz |
| `ptbxl` | PTB-XL | `ppx` | 500 Hz |
| `stpetersburg` | St. Petersburg INCART | `pin` | 257 Hz |
| `zzu_pecg` | ZZU pECG (소아) | `zzu` | 500 Hz |

---

## 사용법

### 1. 전체 변환

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

### 2. 후처리

```bash
# fiducial point/feature 추가
python append_fiducial.py \
    --csv /data/h5/all/v1.0/ecg_table.csv \
    --h5_root /data/h5/all/v1.0 \
    --num_cpus 64

# 신호 품질 추가 (bs_corr, bs_dtw)
python append_signal_quality.py \
    --csv /data/h5/all/v1.0/ecg_table.csv \
    --h5_root /data/h5/all/v1.0 \
    --num_cpus 64

# DTW 생략 (속도 우선)
python append_signal_quality.py \
    --csv /data/h5/all/v1.0/ecg_table.csv \
    --h5_root /data/h5/all/v1.0 \
    --no_dtw --num_cpus 64
```

### 3. 검증

```bash
# 전체 output_root 검증 (데이터셋별 1개 상세 inspect + 일괄 검증)
python verify_h5.py --output_root /data/h5/all/v1.0 --sample 200

# 단일 파일 상세 검증
python verify_h5.py --file /data/h5/all/v1.0/data/psh12340.h5

# 특정 데이터셋만
python verify_h5.py --output_root /data/h5/all/v1.0 --dataset ptbxl,georgia --sample 100
```

### 4. 변환 테스트 (소량 검증)

데이터셋별로 첫 유효 레코드 1개만 변환하여 전체 파이프라인을 빠르게 점검합니다.
종료 시 `ecg_table_test.csv`를 생성하고 `TABLE_COLS` 컬럼 일치 여부를 검증합니다.

```bash
# 그룹 단위
python test_convert.py --group all \
    --heedb_root /data/raw/heedb/ECG \
    --physionet_root /data/raw/physionet.org/files \
    --zzu_root /data/raw/ZZU-pECG \
    --output_root /tmp/h5_test

# 특정 데이터셋 + beat/fiducial 포함
python test_convert.py --dataset georgia,heedb_i0001 \
    --physionet_root ... --heedb_root ... \
    --output_root /tmp/h5_test \
    --compute_beat --compute_fiducial
```

### 5. 일괄 실행

`run_convert.sh` 내 경로를 수정한 후 실행:

```bash
bash run_convert.sh           # 전체 변환 (heedb + physionet + zzu)
bash run_convert.sh heedb     # HEEDB만
bash run_convert.sh physionet # PhysioNet만
bash run_convert.sh zzu       # ZZU만
```

`run_test_verify.sh`로 테스트 + 검증을 한 번에 수행:

```bash
bash run_test_verify.sh           # 전체 (test_convert → verify_h5)
bash run_test_verify.sh heedb     # HEEDB만
```

### 6. 구버전 H5 마이그레이션

code15, mimic4, physionet2021 구버전 H5를 새 포맷으로 변환:

```bash
python convert_old_h5_to_new.py --dataset code15
python convert_old_h5_to_new.py --dataset all --num_cpus 32
```

---

## H5 파일 구조

모든 데이터셋이 동일한 구조를 사용합니다.

```
{file_name}.h5
├── attrs
│   ├── dataset_version           # str   "1.0"
│   ├── file_name                 # str   고유 식별자 ({prefix}{pid}{rid})
│   ├── beat_ext_method           # str   "neurokit2" 또는 ""
│   └── fidu_extract_method       # str   "neurokit2-dwt" 또는 ""
└── ECG/
    ├── metadata/
    │   ├── attrs: record_name, n_sig(=12), fs, sig_len,
    │   │         base_time, base_date, dtype("fp16")
    │   └── datasets: sig_name, fmt, adc_gain, baseline,
    │                 units, adc_res, adc_zero
    └── segments/
        ├── attrs: seg_len            # 세그먼트 개수 (10초 단위 분할)
        ├── 0/
        │   ├── signal              # float16, shape (12, fs*10)
        │   ├── beat_annotation/    # 선택: sample, symbol, subtype, chan, num, aux_note
        │   ├── fiducial_point/     # 선택: fsample, fiducial
        │   └── fiducial_feature/   # 선택: attrs (p_amp, q_amp, ..., 19개)
        ├── 1/  ...
        └── N/  ...
```

### 세그먼트 분할 규칙

- 신호 길이 ≥ 10초인 경우 `fs * 10` 샘플 단위로 잘라 N개 세그먼트로 저장 (`seg_len = N`)
- 끝의 10초 미만 잔여는 버립니다
- 신호 길이 < 10초인 경우 1개 세그먼트로 그대로 저장 (`seg_len = 1`)
- `beat_annotation` / `fiducial_point` / `fiducial_feature`는 세그먼트별로 독립 계산
- `append_fiducial.py`, `append_signal_quality.py`도 모든 세그먼트를 순회하여 처리합니다
  (signal_quality는 세그먼트를 시간축으로 concat한 후 통계를 계산)

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
├── combined_metadata.csv      # HEEDB 양 기관 메타데이터 통합 (HEEDB 포함 시)
└── conversion.log
```

### CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| `filepath` | H5 상대 경로 (`data/xxx.h5`) |
| `dataset` | 데이터셋 키 (예: `ptbxl`, `heedb_i0001`) |
| `pid` | 환자 ID |
| `rid` | 레코드 인덱스 |
| `sid` | 세그먼트 ID (현재 항상 `0` — 파일 단위 1행) |
| `oid` | 고유 관측 ID (`{prefix}{pid}{rid}{sid}`) |
| `age` | 나이 (years/100, 범위 0~1.5) |
| `gender` | 1=남, -1=여, 0=미상 |
| `height` / `weight` | 키/몸무게 (미수집 시 NaN) |
| `fs` | 샘플링 주파수 (Hz) |
| `channel_name` | 채널 순서 문자열 (`TARGET_SIG_NAME` 직렬화) |
| `nan_ratio` ~ `amp_kurtosis` | per-lead 신호 통계 (12-element list, 전 세그먼트 concat 기준) |
| `bs_corr` | per-lead beat-to-beat 상관계수 (**별도 계산**) |
| `bs_dtw` | per-lead beat-to-beat DTW 거리 (**별도 계산**) |

---

## 권장 실행 순서

```
1. 전체 변환       python convert_to_h5.py --group all ...
                    → ecg_table.csv 자동 생성

2. 검증 (선택)     python verify_h5.py --output_root ... --sample 200

3. fiducial 추가   python append_fiducial.py --csv ecg_table.csv ...

4. 신호 품질 추가  python append_signal_quality.py --csv ecg_table.csv ...
                    → bs_corr, bs_dtw 컬럼 추가
```

---

## CLI 옵션 요약

### `convert_to_h5.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--group` | — | `heedb` / `physionet` / `zzu` / `all` |
| `--dataset` | — | 쉼표 구분 데이터셋명 (예: `georgia,ptbxl`) |
| `--heedb_root` | — | HEEDB ECG 루트 (I0001, I0006 상위) |
| `--physionet_root` | — | PhysioNet 데이터 루트 |
| `--zzu_root` | — | ZZU-pECG 루트 |
| `--output_root` | 필수 | H5 출력 루트 |
| `--num_cpus` | 64 | Ray CPU 수 |
| `--batch_size` | 2000 | 배치 크기 |
| `--compute_beat` | OFF | beat_annotation 생성 (후처리 권장) |
| `--compute_fiducial` | OFF | fiducial 생성 (후처리 권장) |

### `append_fiducial.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--csv` | 필수 | 대상 CSV 경로 |
| `--h5_root` | 필수 | H5 루트 |
| `--dataset` | 전체 | 쉼표 구분 데이터셋 필터 |
| `--overwrite` | OFF | 기존 데이터 재계산 |
| `--no_beat` | OFF | beat_annotation 생략 |
| `--no_fiducial` | OFF | fiducial 생략 |
| `--num_cpus` | 32 | Ray CPU 수 |

### `append_signal_quality.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--csv` | 필수 | 대상 CSV 경로 |
| `--h5_root` | 필수 | H5 루트 |
| `--no_dtw` | OFF | DTW 생략 (속도 우선) |
| `--overwrite` | OFF | 기존 값 재계산 |
| `--backup` | OFF | 원본 CSV 백업 |
| `--num_cpus` | 32 | Ray CPU 수 |
| `--save_interval` | 10 | N 배치마다 중간 저장 |

---

## 주의 사항

- **증분 변환**: 이미 존재하는 H5 파일은 자동으로 건너뜁니다. 재변환하려면 파일을 삭제 후 재실행하세요.
- **beat/fiducial**: 변환 시 생성 가능하지만, 속도를 위해 변환 후 `append_fiducial.py`로 별도 실행을 권장합니다.
- **beat_similarity**: 계산 비용이 크므로 변환과 분리하여 `append_signal_quality.py`로 실행합니다. Resume을 지원합니다.
- **MIMIC-IV-ECG**: 변환 전 `python mimic_preprocessing.py`를 먼저 실행해야 합니다.
