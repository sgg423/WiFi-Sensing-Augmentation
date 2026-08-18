# SenseFi Official CSI Experiment Log

## 날짜별 한눈에 보기

| 날짜 | 이날 수행한 작업 | 핵심 결과 | 상태 | 다음 작업 |
|---|---|---|---|---|
| 2026-08-11 | HAR-1 공식 complex CSI 구조 확인; SenseFi in-domain 및 3-fold cross-participant 평가; 환경 메타데이터 점검; SignFi 기반 CSI baseline 구현 및 sanitized-phase 1차 측정 | SenseFi in-domain **94.64%**; cross-participant **14.51±1.69%**; SignFi sanitized-phase in-domain **25.05%** | SenseFi baseline 완료; SignFi 1차 결과는 입력 구조·수렴·phase 처리 확인이 필요한 preliminary baseline | SignFi 학습 history 확인; raw-phase 측정; epoch/normalization ablation 후 최종 SignFi baseline 확정 |
| 2026-08-13 | HAR-3 Classroom M1 공식 CSI 추출 및 in-domain 측정; external-test 기능 구현; HAR-1 Kitchen M1 ↔ HAR-3 Classroom M1 양방향 cross-environment 평가 | HAR-1→HAR-3 **52.64%**; HAR-3→HAR-1 **58.42%**; 양방향 평균 accuracy **55.53%**, Macro-F1 **49.99%**, Macro-recall **58.54%** | SenseFi CSI 양방향 cross-environment real-only baseline 완료 | 각 방향의 동일 test set에서 CSI 증강 전후 비교 |
| 2026-08-18 | RF-Diffusion 생성 HAR-1 M1 CSI 구조 검증; SenseFi synthetic train-only 기능 추가 | 생성 MAT 10,292개/18GB; 각 `(1000,242)` complex64; `cond` 정상; 변환 후 synthetic window 41,168개로 real 대비 **+100%** | Real validation/test 고정 및 synthetic train-only 증강 평가 준비 완료 | 생성 CSI HDF5 변환 → real+synthetic in-domain/cross-environment 측정 |

새로운 작업이나 결과가 나오면 이 표에 날짜별로 한 행씩 추가한다. 실행
명령어와 상세 결과는 아래 날짜별 작업 일지에 기록한다.

## Work log by date

### 2026-08-11

#### 1. Official HAR-1 CSI structure checked — completed

- Official CSI directory:
  `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/CSI/M1_rf_windows_1000`
- Example file: `user0014_w0019.mat`
- MAT variable: `feature`
- Shape and dtype: `(1000, 242)`, `complex64`
- Interpretation: 1,000 time-domain packets and 242 complex CSI features
- SenseFi representation: `abs(CSI)` divided into 250-packet windows
- Final input shape per window: `(1, 250, 242)`

#### 2. Official CSI conversion pipeline updated — completed

Updated `scripts/har1_csi_to_sensefi.py` to support both the previous 90-feature
format and the official 242-feature format. The converter now detects the
feature width automatically and obtains the participant and activity label from
filenames such as `user0014_w0019.mat` when `cond` is unavailable.

Conversion command and output:

```bash
python scripts/har1_csi_to_sensefi.py \
  --input-dir /home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/CSI/M1_rf_windows_1000 \
  --output /home/leehan/datasets/har1_official_csi_242.h5 \
  --window 250 \
  --stride 250 \
  --recursive
```

This command creates the official-CSI SenseFi input dataset. It does not train
or evaluate the sensing model.

#### 3. SenseFi in-domain baseline — completed

Command type: `--split random-window`

Result:

- Accuracy: **94.64%**
- Macro-F1: **94.56%**
- Macro-recall: **94.65%**
- Train/validation/test samples: 28,817 / 6,175 / 6,176
- Seed: 111

This is the **SenseFi-based CSI random-window in-domain baseline**. It is not a
cross-participant result.

#### 4. SenseFi cross-participant baseline — completed

Initial command:

```bash
python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har1_official_csi_242.h5 \
  --output-dir /home/leehan/results/sensefi_har1_official242_cross_user14 \
  --model resnet18 \
  --split participant \
  --test-participant 14 \
  --epochs 50 \
  --seed 111
```

This run was invalid because participant 14 does not exist in the converted
dataset. It produced `test_samples: 0` and `NaN` metrics and must not be included
in any baseline table.

Actual participant sample counts:

| Participant | Samples |
|---:|---:|
| 1 | 14,440 |
| 2 | 12,920 |
| 3 | 13,808 |

Cross-participant evaluation must therefore use held-out participants 1, 2, and
3. Training and validation use the other two participants, and the held-out
participant is used only for the final test. The high `val_accuracy` printed
during training measures validation performance on the training-domain
participants; it is not the held-out-participant result.

#### 5. Held-out participant 1 result — completed

Training/validation participants: 2 and 3
Test participant: 1

```bash
python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har1_official_csi_242.h5 \
  --output-dir /home/leehan/results/sensefi_har1_official242_cross_user1 \
  --model resnet18 \
  --split participant \
  --test-participant 1 \
  --epochs 50 \
  --seed 111
```

| Item | Value |
|---|---:|
| Train samples | 24,055 |
| Validation samples | 2,673 |
| Test samples | 14,440 |
| Accuracy | **15.94%** |
| Macro-F1 | **12.24%** |
| Macro-recall | **14.91%** |

Interpretation: the model trained on participants 2 and 3 classified the 20
activities of unseen participant 1 with 15.94% accuracy. This is a valid
real-only cross-participant fold.

The large difference between the in-domain accuracy (94.64%) and held-out
participant 1 accuracy (15.94%) indicates a substantial participant-domain
shift for the CSI-amplitude SenseFi baseline. This result is retained as a
target condition for the subsequent augmentation experiment.

Valid result location:
`/home/leehan/results/sensefi_har1_official242_cross_user1/result.json`

#### 6. Held-out participants 2 and 3 — completed

| Held-out participant | Training/validation participants | Train | Validation | Test | Accuracy | Macro-F1 | Macro-recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 and 3 | 25,423 | 2,825 | 12,920 | **14.95%** | **12.66%** | **14.16%** |
| 3 | 1 and 2 | 24,624 | 2,736 | 13,808 | **12.64%** | **9.95%** | **12.69%** |

#### 7. Final three-fold cross-participant baseline

| Metric | Participant 1 | Participant 2 | Participant 3 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Accuracy | 15.94% | 14.95% | 12.64% | **14.51 ± 1.69%** |
| Macro-F1 | 12.24% | 12.66% | 9.95% | **11.62 ± 1.46%** |
| Macro-recall | 14.91% | 14.16% | 12.69% | **13.92 ± 1.13%** |

Interpretation: the SenseFi CSI-amplitude model performs well in-domain
(94.64%) but falls to 14.51% mean accuracy for unseen participants. The
80.13-percentage-point gap is evidence of a strong participant-domain shift.
This three-fold mean is the official **real-only cross-participant CSI
baseline** for subsequent augmentation comparisons.

Status: **all three participant folds completed.**

#### 8. Cross-environment metadata inspection — completed

The official-CSI HDF5 currently contains the following domains:

| Metadata | Available values | Samples |
|---|---|---:|
| Participant | 1, 2, 3 | 14,440 / 12,920 / 13,808 |
| Day | 1 only | 41,168 |
| Monitor | 1 only | 41,168 |

Every sample belongs to `day=1` and `monitor=1`. Therefore, this HDF5 cannot
produce a valid cross-day or cross-monitor split. A cross-environment run on the
current file would have either no held-out test samples or no training samples.

Additional HAR-1 source folders containing different days, monitors, locations,
or environments must be identified and converted before cross-environment
evaluation. Environment domains must be separated before window generation and
must not be mixed through a random-window split.

Status: **cross-environment evaluation pending additional environment-domain
data.**

#### 9. SignFi-based CSI baseline pipeline — implementation completed

The next CSI baseline follows the SignFi example used as the CSI comparison
family in BeamSense. Official HAR-1 complex CSI is converted into amplitude and
phase. The two components are concatenated along image width, following the
official SignFi input preparation.

| Item | HAR-1 adaptation |
|---|---|
| Original complex CSI window | `(250,242)` |
| SignFi amplitude | `(250,242)` |
| SignFi phase | raw or unwrapped/linear-trend-removed `(250,242)` |
| CNN input | `(1,250,484)` |
| Architecture | Conv `4×4` (4 filters) → BN → ReLU → MaxPool `4×4` → FC |
| Output adaptation | Original 276 signs → HAR-1 20 activities |
| Optimizer | SGD, learning rate 0.01, momentum 0.9, L2 0.01 |

Implemented files:

- `scripts/har1_csi_to_signfi.py`
- `scripts/train_signfi_har1.py`

Two input variants will be reported separately:

1. `raw`: CSI amplitude plus raw phase.
2. `sanitized`: CSI amplitude plus unwrapped phase after removing the
   per-packet linear phase trend over subcarriers. This is the primary variant
   corresponding to the phase-preprocessing comparison discussed by BeamSense.

Status: **implementation complete; sanitized-phase in-domain measurement
completed; additional diagnostic runs pending.**

#### 10. SignFi sanitized-phase in-domain result — preliminary

| Item | Value |
|---|---:|
| Train samples | 28,817 |
| Validation samples | 6,175 |
| Test samples | 6,176 |
| Accuracy | **25.05%** |
| Macro-F1 | **22.31%** |
| Macro-recall | **24.26%** |
| Split | Random-window in-domain |
| Epochs | 10 |
| Seed | 111 |

This result is above the 5% chance level for 20 classes but is substantially
below the SenseFi baseline. It is retained as a preliminary result rather than
discarded. Before defining it as the final SignFi baseline, the following must
be checked:

1. Whether validation accuracy was still increasing at epoch 10.
2. Raw phase versus sanitized phase under the same split.
3. Official SignFi zero-centering versus the current train-global
   per-component min-max normalization.
4. The effect of adapting SignFi from its original three-antenna input to the
   single-link HAR-1 CSI input.

Status: **preliminary result recorded; diagnostic and ablation runs pending.**

### 2026-08-13

#### 1. HAR-3 Classroom M1 official CSI extraction — completed

- Source directory:
  `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/CSI/M1_rf_windows_1000`
- Environment: Classroom
- Device/link identifier: M1
- Propagation condition: LoS
- Target comparison: HAR-1 Kitchen M1 → HAR-3 Classroom M1

#### 2. SenseFi external-environment evaluation support — completed

`scripts/train_sensefi_har1.py` now supports `--split external` and
`--test-data`. Training and validation are created only from the source HDF5,
while every sample in the external HDF5 is reserved for testing. Source-train
normalization parameters are also applied to the external test set, preventing
target-environment normalization leakage.

Status: **implementation complete; HAR-3 conversion and GPU evaluation
in-domain and HAR-1 → HAR-3 cross-environment evaluations completed.**

#### 3. HAR-3 Classroom M1 in-domain baseline — completed

| Metric | Result |
|---|---:|
| Accuracy | **97.75%** |
| Macro-F1 | **97.37%** |
| Macro-recall | **97.37%** |
| Split | Random-window in-domain |
| Model | SenseFi ResNet18, 20 classes |
| Seed | 111 |

Interpretation: HAR-3 official CSI supports highly accurate 20-class activity
recognition under the same-domain random-window setting. Together with the
HAR-1 in-domain accuracy of 94.64%, this confirms that both source and target
datasets, labels, CSI extraction, and SenseFi preprocessing pipelines are
operational. A low HAR-1 → HAR-3 result can therefore be interpreted primarily
as an environment-domain shift rather than a broken HAR-3 dataset.

#### 4. HAR-1 Kitchen M1 → HAR-3 Classroom M1 — completed

| Metric | Result |
|---|---:|
| Accuracy | **52.64%** |
| Macro-F1 | **54.83%** |
| Macro-recall | **59.29%** |
| Source train/validation | HAR-1 Kitchen M1 |
| External test | HAR-3 Classroom M1, all samples |
| Model | SenseFi ResNet18, 20 classes |
| Seed | 111 |

The model was trained and normalized only with HAR-1 source-domain data. HAR-3
was reserved entirely for the final external test. Accuracy decreased by 42.00
percentage points relative to the HAR-1 in-domain result (94.64%) and by 45.11
percentage points relative to the HAR-3 in-domain result (97.75%). Since both
datasets achieve high in-domain accuracy, this degradation provides evidence of
a substantial Kitchen-to-Classroom environment-domain shift.

This value is the **real-only CSI cross-environment baseline** for the
HAR-1 → HAR-3 direction. The subsequent augmented experiment must keep the
HAR-3 test set unchanged and add synthetic data only to source-domain training.

#### 5. HAR-3 Classroom M1 → HAR-1 Kitchen M1 — completed

| Metric | Result |
|---|---:|
| Accuracy | **58.42%** |
| Macro-F1 | **45.15%** |
| Macro-recall | **57.79%** |
| Source train/validation | HAR-3 Classroom M1 |
| External test | HAR-1 Kitchen M1, all samples |
| Model | SenseFi ResNet18, 20 classes |
| Seed | 111 |

#### 6. Bidirectional cross-environment summary

| Direction | Accuracy | Macro-F1 | Macro-recall |
|---|---:|---:|---:|
| HAR-1 Kitchen → HAR-3 Classroom | 52.64% | 54.83% | 59.29% |
| HAR-3 Classroom → HAR-1 Kitchen | 58.42% | 45.15% | 57.79% |
| Direction-level mean | **55.53%** | **49.99%** | **58.54%** |

The reverse direction is 5.78 percentage points higher in accuracy, showing
that cross-environment transfer is asymmetric. However, its Macro-F1 is lower
than in the forward direction, indicating that the higher overall accuracy may
be concentrated in a subset of activity classes. Per-class recall and the
confusion matrix should be inspected before claiming that HAR-3 → HAR-1 is
uniformly easier.

The direction-level mean of 55.53% is reported only as a compact summary. Each
direction remains a separate baseline because the source training set and test
distribution differ.

### 2026-08-14

#### 1. M1-based BeamSense-style SignFi evaluation — implementation completed

The SignFi evaluation is corrected to match the CSI comparison pipeline used by
BeamSense rather than treating the original one-layer SignFi example CNN as the
BeamSense comparison classifier.

| Component | M1-based implementation |
|---|---|
| CSI source | Official HAR-1/HAR-3 M1 complex CSI |
| Window | 10 CSI frames, non-overlapping |
| Preprocessing | Amplitude + unwrapped, linear-trend-removed phase |
| Input | Single-monitor `(1,10,484)` |
| Classifier | BeamSense Fig. 7 VGG: three two-convolution blocks, 128→64→32 filters, MaxPool, FC |
| Evaluation | Random-window in-domain and external cross-environment |

This is explicitly named a **single-M1-monitor adaptation**. BeamSense's
reported CSI comparison used M1, M2, and M3 monitors, so its reported 90–93%
preprocessed-CSI accuracy is not an expected reproduction target for the M1-only
experiment.

The previously measured 25.05% remains recorded as an original SignFi example
CNN adaptation and is excluded from the BeamSense-style SignFi baseline.

Status: **code complete; HAR-1/HAR-3 10-frame conversion and GPU evaluation
pending.**

### 2026-08-18

#### 1. RF-Diffusion-generated HAR-1 M1 CSI prepared

- Generated MAT directory: `/mnt/ssd1/leehan/M1_rf_generated_1000`
- Intended use: synthetic source-domain training data only
- Prohibited use: validation, HAR-1 real test, and HAR-3 external test

Generated-file validation:

| Item | Verified value |
|---|---|
| MAT files | 10,292 |
| Disk size | 18 GB |
| `feature` | `(1000,242)`, `complex64` |
| `cond` | `(1,4)`, `[activity, day, monitor, participant]` |
| Example first condition | `[1,1,1,1]` |
| Example final condition | `[20,1,1,3]` |
| Expected 250-packet windows | 41,168 |
| Augmentation ratio | **+100% synthetic relative to the full real-window count** |

The generated filenames use sample/window indices, while activity and domain
labels are read from `cond`. Therefore, filename tokens such as `w0000` are not
used as activity labels.

#### 2. Leakage-safe SenseFi augmentation input — implementation completed

`scripts/train_sensefi_har1.py` now accepts `--augment-data`. The real-data
split is generated with the same seed as the real-only baseline. Synthetic
samples are appended only to the real training split, while validation and test
loaders remain real-only. Min-max normalization is calculated from the real
training split and reused for synthetic and test samples.

The `--augment-ratio` option defines the synthetic/real-training sample ratio.
The default `1.0` selects a class-stratified synthetic subset equal in size to
the actual real training split (+100%). This prevents the full 41,168-sample
synthetic pool from unintentionally becoming +143% in the in-domain 70% train
split or about +111% in the external 90% train split.

This design allows a direct comparison:

```text
real-only baseline:      real source train → fixed real validation/test
augmentation condition: real source train + synthetic source train
                        → same fixed real validation/test
```

Status: **training support and generated-file validation complete; HDF5
conversion and accuracy measurement pending.**

---

The sections below contain the full commands, result definitions, and reporting
rules for reproducing each experiment.

## 1. Experiment objective

Measure a SenseFi-based CSI sensing baseline on the official CSI extracted from
the CSI-BFI-HAR HAR-1 dataset. The sensing task is 20-class human activity
recognition.

This experiment uses the official SenseFi UT-HAR ResNet18 architecture with its
output layer changed from 7 to 20 classes. The model is trained from scratch on
HAR-1; this is not a reproduction of the original SenseFi accuracy.

## 2. Data and preprocessing

- Source dataset: CSI-BFI-HAR / HAR-1
- Official CSI directory:
  `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/CSI/M1_rf_windows_1000`
- Example file: `user0014_w0019.mat`
- MAT variable: `feature`
- Original sample shape: `(1000, 242)`
- Original dtype: `complex64`
- Representation used by SenseFi: `abs(CSI)`
- Window size and stride: 250 packets / 250 packets
- SenseFi input shape per window: `(1, 250, 242)`
- Label example: `w0019` is converted to zero-based class index `18`

### CSI-to-SenseFi conversion command

```bash
cd /home/leehan/RF-Diffusion

python scripts/har1_csi_to_sensefi.py \
  --input-dir /home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/CSI/M1_rf_windows_1000 \
  --output /home/leehan/datasets/har1_official_csi_242.h5 \
  --window 250 \
  --stride 250 \
  --recursive
```

Output dataset:
`/home/leehan/datasets/har1_official_csi_242.h5`

## 3. In-domain baseline

### Meaning

The `random-window` split randomly separates CSI windows into training,
validation, and test sets. The same participants and measurement distribution
can occur in both training and test sets. Therefore, this result is reported as
a **random-window in-domain baseline**, not as cross-user generalization.

### Command

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har1_official_csi_242.h5 \
  --output-dir /home/leehan/results/sensefi_har1_official242_random \
  --model resnet18 \
  --split random-window \
  --epochs 50 \
  --seed 111
```

### Result

| Item | Value |
|---|---:|
| Train samples | 28,817 |
| Validation samples | 6,175 |
| Test samples | 6,176 |
| Accuracy | **94.64%** |
| Macro-F1 | **94.56%** |
| Macro-recall | **94.65%** |
| Normalization | Train-global min-max |
| Seed | 111 |

Interpretation: SenseFi ResNet18 correctly classified 94.64% of the official
HAR-1 CSI test windows under the random-window in-domain setting. This value is
the **real-only CSI baseline** for the same evaluation protocol.

Raw result file:
`/home/leehan/results/sensefi_har1_official242_random/result.json`

## 4. Cross-participant baseline

### Meaning

The `participant` split excludes one participant completely from training and
validation. All samples from that participant are used only for testing. This
measures generalization to an unseen participant and is reported as a
**cross-participant (cross-domain) baseline**.

### Check participant identifiers

```bash
python -c "import h5py; p='/home/leehan/datasets/har1_official_csi_242.h5'; f=h5py.File(p,'r'); print(sorted(set(f['participant'][:].tolist()))); f.close()"
```

### Command for one held-out participant

The following completed example holds out participant 1.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har1_official_csi_242.h5 \
  --output-dir /home/leehan/results/sensefi_har1_official242_cross_user1 \
  --model resnet18 \
  --split participant \
  --test-participant 1 \
  --epochs 50 \
  --seed 111
```

### Results to record

| Held-out participant | Accuracy | Macro-F1 | Macro-recall | Result path |
|---:|---:|---:|---:|---|
| 1 | 15.94% | 12.24% | 14.91% | `/home/leehan/results/sensefi_har1_official242_cross_user1/result.json` |
| 2 | 14.95% | 12.66% | 14.16% | `/home/leehan/results/sensefi_har1_official242_cross_user2/result.json` |
| 3 | 12.64% | 9.95% | 12.69% | `/home/leehan/results/sensefi_har1_official242_cross_user3/result.json` |
| Mean ± sample SD | **14.51 ± 1.69%** | **11.62 ± 1.46%** | **13.92 ± 1.13%** | Three-fold summary |

All participants were evaluated. The mean and sample standard deviation are
reported separately from the random-window in-domain result.

## 5. Result terminology

| Result | Correct name | What it measures |
|---|---|---|
| `--split random-window` | SenseFi-based CSI in-domain baseline | Classification within the same participant/environment distribution |
| `--split participant` | SenseFi-based CSI cross-participant baseline | Generalization to a participant excluded from training |
| Real + synthetic training | CSI augmentation result | Performance after adding generated CSI only to the training set |
| Augmented accuracy minus real-only accuracy | CSI augmentation gain | Improvement attributable to the augmentation experiment |

The CSI and BFI raw accuracies should not by themselves be interpreted as an
intrinsic modality ranking because their input representations and sensing
models differ. The primary augmentation comparison is the performance change
from each modality's own real-only baseline.
