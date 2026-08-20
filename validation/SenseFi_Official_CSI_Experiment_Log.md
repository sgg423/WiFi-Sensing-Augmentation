# SenseFi Official CSI Experiment Log

## 날짜별 한눈에 보기

| 날짜 | 이날 수행한 작업 | 핵심 결과 | 상태 | 다음 작업 |
|---|---|---|---|---|
| 2026-08-11 | HAR-1 공식 complex CSI 구조 확인; SenseFi in-domain 및 3-fold cross-participant 평가; 환경 메타데이터 점검; SignFi 기반 CSI baseline 구현 및 sanitized-phase 1차 측정 | SenseFi in-domain **94.64%**; cross-participant **14.51±1.69%**; SignFi sanitized-phase in-domain **25.05%** | SenseFi baseline 완료; SignFi 1차 결과는 입력 구조·수렴·phase 처리 확인이 필요한 preliminary baseline | SignFi 학습 history 확인; raw-phase 측정; epoch/normalization ablation 후 최종 SignFi baseline 확정 |
| 2026-08-13 | HAR-3 Classroom M1 공식 CSI 추출 및 in-domain 측정; external-test 기능 구현; HAR-1 Kitchen M1 ↔ HAR-3 Classroom M1 양방향 cross-environment 평가 | HAR-1→HAR-3 **52.64%**; HAR-3→HAR-1 **58.42%**; 양방향 평균 accuracy **55.53%**, Macro-F1 **49.99%**, Macro-recall **58.54%** | SenseFi CSI 양방향 cross-environment real-only baseline 완료 | 각 방향의 동일 test set에서 CSI 증강 전후 비교 |
| 2026-08-18 | RF-Diffusion 생성 HAR-1 M1 CSI 구조 검증; SenseFi 증강 평가; 512-packet M1/HAR-3 변환 및 baseline 측정 | ResNet18 250-packet in-domain **94.64% → 96.10% (+1.46%p)**, cross-environment **52.64% → 54.06% (+1.42%p)**; LeNet **90.98% → 71.15% (-19.84%p)**; 512-packet M1 **90.72%**, HAR-3 **98.47%** | 두 환경의 512-packet in-domain 기준값 확보; window 길이보다 환경·데이터 구성의 영향 확인 | M1→HAR-3 512-packet cross-environment baseline 측정 |
| 2026-08-19 | HAR-1 M1 BFI 공식 추출; 10-frame BeamSense 데이터셋 및 real-only baseline 측정 | Random-window accuracy **91.55%**; held-out P1 accuracy **10.26%**, Macro-F1 **6.35%**, Macro-recall **8.99%** | In-domain baseline 확보; 단일-M1 zero-shot cross-participant에서 큰 domain gap 확인 | P2/P3 held-out fold 측정 후 3-fold 평균; BFI 증강 전후 평가 |
| 2026-08-20 | HAR-3 RF-Diffusion 증강 평가; limited-real 및 source-file 제한 실험 | 전체 real **97.75% → 98.39%**; real 10% **89.73%**; source-file 전체 **97.26%**; 5 source/class **45.74%** | 클래스당 source file을 5개로 제한하면 독립 source-file test 성능이 크게 하락 | 10/20 source baseline 및 동일 source만 사용한 RF-Diffusion 재학습·증강 평가 |

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

#### 3. HAR-1 M1 in-domain +100% augmentation — completed

Training command:

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har1_official_csi_242.h5 \
  --augment-data /mnt/ssd1/leehan/har1_generated_csi_sensefi_242.h5 \
  --augment-ratio 1.0 \
  --output-dir /home/leehan/results/sensefi_har1_aug100_random \
  --model resnet18 \
  --split random-window \
  --epochs 50 \
  --seed 111
```

Data composition and result:

| Item | Value |
|---|---:|
| Real training samples | 28,817 |
| Synthetic training samples | 28,817 |
| Real : synthetic training ratio | 1 : 1 |
| Total training samples | 57,634 |
| Real validation samples | 6,175 |
| Real test samples | 6,176 |
| Accuracy | **96.10%** |
| Macro-F1 | **96.00%** |
| Macro-recall | **95.94%** |

Comparison against the real-only HAR-1 M1 in-domain baseline:

| Metric | Real only | Real + synthetic | Absolute change |
|---|---:|---:|---:|
| Accuracy | 94.64% | **96.10%** | **+1.46%p** |
| Macro-F1 | 94.56% | **96.00%** | **+1.44%p** |
| Macro-recall | 94.65% | **95.94%** | **+1.28%p** |

The error rate decreased from approximately 5.36% to 3.90%, corresponding to
a relative error reduction of approximately **27.2%**. This result demonstrates
a positive in-domain augmentation effect under the evaluated 1:1 training
condition. It does not yet establish cross-environment improvement; that must be
measured separately using the fixed real HAR-3 external test set.

#### 4. HAR-1 M1 → HAR-3 M1 cross-environment +100% augmentation — completed

Training and evaluation command:

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har1_official_csi_242.h5 \
  --augment-data /mnt/ssd1/leehan/har1_generated_csi_sensefi_242.h5 \
  --augment-ratio 1.0 \
  --test-data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --output-dir /home/leehan/results/sensefi_har1_aug100_to_har3 \
  --model resnet18 \
  --split external \
  --epochs 50 \
  --seed 111
```

Data composition and result:

| Item | Value |
|---|---:|
| HAR-1 real training samples | 37,051 |
| HAR-1 synthetic training samples | 37,051 |
| Real : synthetic training ratio | 1 : 1 |
| Total training samples | 74,102 |
| HAR-1 real validation samples | 4,117 |
| HAR-3 real external-test samples | 48,860 |
| Accuracy | **54.06%** |
| Macro-F1 | **58.35%** |
| Macro-recall | **62.13%** |

Comparison against the real-only HAR-1 M1 → HAR-3 M1 baseline:

| Metric | Real only | Real + synthetic | Absolute change |
|---|---:|---:|---:|
| Accuracy | 52.64% | **54.06%** | **+1.42%p** |
| Macro-F1 | 54.83% | **58.35%** | **+3.52%p** |
| Macro-recall | 59.29% | **62.13%** | **+2.84%p** |

The fixed HAR-3 real test set was excluded from both training and normalization.
Therefore, the positive changes provide evidence that the generated HAR-1 CSI
improves not only same-domain recognition but also transfer from the Kitchen
environment to the Classroom environment. The gains are preliminary single-run
results with seed 111; multiple seeds or repeated trials are required to report
statistical significance.

Status: **HAR-1 CSI in-domain and HAR-1 → HAR-3 cross-environment +100%
augmentation evaluations completed. Both improved over their real-only
baselines.**

#### 5. SenseFi LeNet HAR-1 input adaptation — completed

The published UT-HAR LeNet assumes an encoder output of `(96,4,4)`. With the
official HAR-1 `(1,250,242)` input, the encoder instead produces `(96,4,13)`.
The previous fixed `view(-1, 96*4*4)` therefore changed a batch of 64 into 208
predictions and caused `Expected input batch_size (208) to match target
batch_size (64)`.

An adaptive `4x4` average-pooling layer was added after the original convolution
encoder. This preserves the published fully connected input size and the real
batch dimension while allowing the 242-subcarrier HAR-1 representation. The
forward pass now uses `torch.flatten(x, 1)` so the batch dimension cannot be
silently inferred incorrectly.

LeNet real-only in-domain result (seed 111):

| Metric | Result |
|---|---:|
| Accuracy | **90.98%** |
| Macro-F1 | **90.84%** |
| Macro-recall | **90.88%** |

The random-window split, real train/validation/test sample counts, normalization,
and seed are identical to the corresponding ResNet18 real-only experiment. This
result is the LeNet reference point for the subsequent 1:1 synthetic
augmentation comparison.

LeNet 1:1 real/synthetic in-domain result (seed 111):

| Metric | Real only | Real + synthetic | Absolute change |
|---|---:|---:|---:|
| Accuracy | 90.98% | **71.15%** | **-19.84%p** |
| Macro-F1 | 90.84% | **70.45%** | **-20.39%p** |
| Macro-recall | 90.88% | **70.64%** | **-20.24%p** |

Unlike ResNet18, LeNet was substantially degraded by the 1:1 synthetic mixture.
The generated CSI is therefore not model-independent under the current mixing
condition. A plausible interpretation is that the lower-capacity LeNet and its
adaptive spatial compression are more sensitive to the real/synthetic domain
gap. This interpretation must be tested with lower augmentation ratios rather
than asserted from a single ratio and seed.

Status: **LeNet real-only and +100% augmented in-domain evaluations completed;
lower augmentation ratios and both cross-environment runs pending.**

#### 6. M1 512-packet ResNet18 in-domain baseline — completed

The pre-windowed M1 directory contained 21,143 MAT files with complex
`feature` shape `(512,242)`. The converter accepted 20,188 samples and skipped
955 files whose activity/user metadata did not satisfy the 20-class rule.

| Item | Value |
|---|---:|
| Converted samples | 20,188 |
| Real training samples | 14,131 |
| Real validation samples | 3,028 |
| Real test samples | 3,029 |
| Input shape | `(1,512,242)` |
| Accuracy | **90.72%** |
| Macro-F1 | **90.84%** |
| Macro-recall | **90.70%** |

This is the real-only, random-window M1 baseline for the 512-packet condition.
It must not be treated as a direct replacement for the earlier 94.64%
250-packet baseline because the window length, sample set, and source-window
construction differ. The skipped-file label distribution and possible
`parent_trace` overlap should be audited before final publication reporting.

Status: **M1 512-packet in-domain baseline completed; HAR-3 512-packet
in-domain and M1 → HAR-3 512-packet external baselines pending.**

#### 7. HAR-3 512-packet ResNet18 in-domain baseline — completed

| Item | Value |
|---|---:|
| Converted samples | 23,946 |
| Real training samples | 16,762 |
| Real validation samples | 3,592 |
| Real test samples | 3,592 |
| Input shape | `(1,512,242)` |
| Accuracy | **98.47%** |
| Macro-F1 | **98.24%** |
| Macro-recall | **98.23%** |

The HAR-3 512-packet result is 7.75 percentage points higher than the M1
512-packet result under the same model and split type. Because HAR-3 reaches a
high score with the 512-packet input, the lower M1 result cannot be attributed
to sequence length alone. Environment-specific class separability, sample
composition, metadata exclusions, and source-trace splitting remain possible
factors. As with M1, this random-window result may contain windows sharing a
`parent_trace` across splits.

Status: **M1 and HAR-3 512-packet in-domain baselines completed; M1 → HAR-3
512-packet external baseline pending.**

### 2026-08-19

#### 1. Official Wi-BFI extraction smoke test — completed

- Source: HAR-1 Kitchen, day 1, monitor M1, activity A, participant P1
- PCAP: `A_1_M1_P1.pcapng`
- Standard/mode/configuration: IEEE 802.11ac, MU, 3x1, 80 MHz
- Beamformee MAC: `b0:b9:8a:63:55:9c`
- Valid MU BFI frames in the trace: 10,226
- Smoke-test frames processed: 200

Official Wi-BFI output validation:

| Output | Shape | Dtype | Complex | Finite | Range/mean |
|---|---|---|---|---|---|
| Reconstructed V | `(200,234,3,1)` | `complex128` | Yes | Yes | magnitude 0.00457–0.99906, mean 0.52985 |
| BFA | `(200,234,4)` | `int64` | No | Yes | 0–511 |

The local extraction chain is operational:

```text
raw PCAPNG → tshark/PyShark MU-frame filtering → four BF angles per
subcarrier → reconstructed complex V matrix
```

Python 3.11 was used because the current PyShark release failed under Python
3.14 when creating the asyncio event loop. The test trace can produce ten
non-overlapping 1000-frame windows with 226 frames left over.

Status: **official BFI extraction validated on one trace; full-trace extraction,
window conversion, and 60-trace HAR-1 M1 batch processing pending.**

#### 2. Full `A_1_M1_P1` BFI trace extraction — completed

| Output | Full shape | Dtype | Finite |
|---|---|---|---|
| Reconstructed V | `(10226,234,3,1)` | `complex128` | Yes |
| BFA | `(10226,234,4)` | `int64` | Yes |

The trace yields ten non-overlapping 1000-frame windows; the final 226 frames
are excluded. PyShark emitted a `TSharkCrashException` with return code 255 only
while cleaning up its child process after extraction. Both output files were
already saved with the complete expected frame count and valid finite values,
so this event is recorded as a shutdown warning rather than data-extraction
failure.

Status: **single-trace full extraction completed; 1000-frame MAT window
conversion and batch extraction pending.**

#### 3. HAR-1 M1 raw BFI set downloaded — completed

- Local directory: `BFI-pcap-all/HAR-1/BFI/M1`
- Expected composition: 20 activities (A–T) × 3 participants (P1–P3)
- PCAPNG files: 60
- Total size: 438 MB

Status: **all HAR-1 M1 PCAPNG inputs acquired; per-trace valid-frame/MAC audit
and resumable batch extraction pending.**

#### 4. HAR-1 M1 per-trace frame audit — 1000-frame class-coverage issue found

- Total non-overlapping 1000-frame windows across 60 traces: 380
- Traces below 1000 valid MU BFI frames:
  - `J_1_M1_P2.pcapng`: 660
  - `S_1_M1_P1.pcapng`: 111
  - `S_1_M1_P2.pcapng`: 23
  - `S_1_M1_P3.pcapng`: 101

All three participant traces for activity S are shorter than 1000 frames.
Consequently, strict non-overlapping 1000-frame conversion would remove the S
class entirely and produce a 19-class dataset. Such a result is not directly
comparable with the 20-class CSI experiment. Zero-padding or repeating 23–111
frames to 1000 may introduce a strong artificial class cue and is not accepted
as the primary solution.

Status: **batch extraction paused before wasting computation; additional valid
S traces or a justified shorter BFI generation window is required for a
20-class experiment.**

#### 5. Official BFA to BeamSense 10-frame dataset — completed

- Official Wi-BFI BFA traces processed: 60
- BeamSense input shape per sample: `(10,234,4)`
- Total non-overlapping samples: 40,675
- Activity coverage: A–T, all 20 classes
- Typical per-class count: 1,639–2,733
- Activity S count: 23

The 10-frame representation restores formal 20-class coverage, but activity S
remains severely underrepresented because its three source traces contain only
111, 23, and 101 valid BFI frames. Balanced class weights can support a
preliminary baseline, but they do not create independent S observations.
Accuracy must therefore be reported together with Macro-F1, Macro-recall, and
the normalized confusion matrix. The limitation must remain explicit until
additional valid S traces are acquired or a paired 19-class protocol is adopted
for both CSI and BFI.

Status: **HAR-1 M1 BeamSense input dataset completed; preliminary real-only
baseline ready, final class-balanced protocol pending S-data decision.**

#### 6. BeamSense real-only random-window baseline — completed

| Item | Value |
|---|---:|
| Train samples | 28,529 |
| Validation samples | 6,064 |
| Test samples | 6,082 |
| Accuracy | **91.55%** |
| Macro-F1 | **91.90%** |
| Macro-recall | **91.81%** |
| Normalization | none |
| Class weighting | balanced |
| Seed | 111 |

This is a BeamSense-style BFA real-only **preliminary in-domain baseline**. The
70/15/15 split is performed at the 10-frame window level, so windows from the
same PCAP trace can occur in both train and test sets. Activity S has only 23
total windows; its test count and normalized confusion row must be reported
separately before interpreting the macro metrics. The result is suitable as the
fixed random-window reference for an identically split augmentation experiment,
but not as evidence of trace- or participant-independent generalization.

S-class audit: only **2** of the 6,082 test windows are activity S, and both were
classified correctly (reported S recall 1.0). This two-sample result is not a
statistically meaningful 100% recall estimate and may benefit from windows of
the same source traces appearing in training. It must not be highlighted as
evidence of S-class generalization. The other test classes contain 227–407
windows each.

Status: **BeamSense random-window baseline completed; S-class audit,
participant-held-out baselines, and synthetic-train-only support pending.**

#### 7. BeamSense cross-participant P1 fold — completed

| Item | Value |
|---|---:|
| Training participants | P2, P3 |
| Held-out test participant | P1 |
| Train / validation / test | 21,723 / 2,397 / 16,555 |
| Accuracy | **10.26%** |
| Macro-F1 | **6.35%** |
| Macro-recall | **8.99%** |
| Random-chance accuracy | 5.00% |

This is a zero-shot, single-M1-monitor participant transfer result: no P1
window is used for training or validation. The large drop from the 91.55%
random-window result demonstrates strong subject dependence and also shows how
the random-window protocol can overstate unseen-subject performance. It is not
a reproduction of BeamSense's adaptation-based or multi-monitor cross-subject
setting. P2 and P3 held-out folds are required before reporting a summary.

### 2026-08-20

#### 1. HAR-3 RF-Diffusion generated CSI conversion — completed

RF-Diffusion으로 생성한 HAR-3 M1 complex CSI를 SenseFi 입력 형식으로
변환했다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/har1_csi_to_sensefi.py \
  --input-dir /mnt/ssd1/leehan/HAR-3_rf_generated_1000 \
  --output /mnt/ssd1/leehan/har3_generated_csi_sensefi_242.h5 \
  --window 250 \
  --stride 250 \
  --recursive
```

| Item | Value |
|---|---:|
| Output HDF5 | `/mnt/ssd1/leehan/har3_generated_csi_sensefi_242.h5` |
| `x` shape | `(48860, 1, 250, 242)` |
| `x` dtype | `float16` |
| `y` shape | `(48860,)` |
| Classes | 0–19 (20 classes) |

#### 2. HAR-3 real + synthetic 1:1 in-domain evaluation — completed

실제 HAR-3 학습 데이터에 RF-Diffusion 생성 데이터를 1:1로 추가했다.
합성 데이터는 학습에만 사용했고 validation/test에는 실제 CSI만 사용했다.
따라서 real-only baseline과 augmentation 결과의 test set은 동일하다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --augment-data /mnt/ssd1/leehan/har3_generated_csi_sensefi_242.h5 \
  --augment-ratio 1.0 \
  --output-dir /home/leehan/results/sensefi_har3_aug100_random \
  --model resnet18 \
  --split random-window \
  --epochs 50 \
  --seed 111
```

| Data partition | Samples |
|---|---:|
| Real train | 34,202 |
| Synthetic train | 34,202 |
| Total train | 68,404 |
| Real validation | 7,329 |
| Real test | 7,329 |
| Available synthetic | 48,860 |

| Metric | Real-only baseline | Real + synthetic (1:1) | Change |
|---|---:|---:|---:|
| Accuracy | 97.7487% | **98.3900%** | **+0.6413%p** |
| Macro-F1 | 97.3656% | **98.1217%** | **+0.7561%p** |
| Macro-recall | 97.3682% | **98.1234%** | **+0.7551%p** |

Accuracy error rate는 2.2513%에서 1.6100%로 감소했으며, baseline error의
**28.48% 상대 감소**에 해당한다. 절대 accuracy 증가는 0.64%p이지만,
baseline이 이미 97.75%로 포화에 가까웠다는 점을 고려하면 오분류의 약
28.5%를 줄인 결과다. HAR-1에서 확인한 +1.46%p와 함께 두 CSI 환경 모두
ResNet18에서 RF-Diffusion 증강 후 성능이 상승했다.

이 결과는 HAR-3 random-window in-domain 조건의 증강 효과이며,
cross-environment 일반화 성능을 의미하지 않는다.

Raw result file:
`/home/leehan/results/sensefi_har3_aug100_random/result.json`

Status: **HAR-3 CSI in-domain augmentation comparison completed.**

#### 3. HAR-3 limited-real 10% baseline — completed

전체 real train split 중 클래스별 10%만 사용해 데이터 부족 조건의
real-only baseline을 측정했다. 최초 70/15/15 random-window split을 먼저
고정한 후 train split만 축소했기 때문에 validation/test 7,329개는 앞선
HAR-3 전체 데이터 실험과 동일하다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --real-train-ratio 0.1 \
  --output-dir /home/leehan/results/sensefi_har3_real10_random \
  --model resnet18 \
  --split random-window \
  --epochs 50 \
  --seed 111
```

| Item | Value |
|---|---:|
| Available real train | 34,202 |
| Real train used | 3,420 (10%) |
| Validation / test | 7,329 / 7,329 |
| Accuracy | **89.7257%** |
| Macro-F1 | **88.3879%** |
| Macro-recall | **88.4240%** |
| Normalization | Train-global min-max |
| Seed | 111 |

전체 real train baseline 97.7487%와 비교하면 accuracy가 8.0229%p 낮아져
ceiling effect가 완화됐다. 이 값은 증강 효과가 아니라 real data를 10%로
줄였을 때의 기준 성능이다. 다음 실험에서는 이때 선택된 동일한 real
3,420개에 synthetic 3,420개만 추가하고 실제 validation/test는 그대로
유지해야 한다.

Raw result file:
`/home/leehan/results/sensefi_har3_real10_random/result.json`

Status: **real 10% baseline and matching 1:1 augmentation run completed.**

#### 4. HAR-3 limited-real 10% + synthetic 1:1 — completed

앞선 baseline에서 사용한 동일한 real train 3,420개에 synthetic CSI
3,420개를 추가했다. Seed, validation/test split, normalization 방식 및 모델은
baseline과 동일하다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --real-train-ratio 0.1 \
  --augment-data /mnt/ssd1/leehan/har3_generated_csi_sensefi_242.h5 \
  --augment-ratio 1.0 \
  --output-dir /home/leehan/results/sensefi_har3_real10_aug100_random \
  --model resnet18 \
  --split random-window \
  --epochs 50 \
  --seed 111
```

| Data partition | Samples |
|---|---:|
| Real train | 3,420 |
| Synthetic train | 3,420 |
| Total train | 6,840 |
| Real validation | 7,329 |
| Real test | 7,329 |

| Metric | Real 10% only | Real 10% + synthetic 1:1 | Change |
|---|---:|---:|---:|
| Accuracy | 89.7257% | **86.5602%** | **-3.1655%p** |
| Macro-F1 | 88.3879% | **84.9473%** | **-3.4406%p** |
| Macro-recall | 88.4240% | **84.9460%** | **-3.4780%p** |

Accuracy error rate는 10.2743%에서 13.4398%로 증가했으며, baseline 대비
오류가 **30.81% 상대 증가**했다. 따라서 현재 생성 CSI를 real 10%와 같은
수로 추가하는 방식은 제한된 실제 데이터를 보완하지 못했다. 전체 real
조건의 +0.64%p와 반대되는 결과이므로, RF-Diffusion 증강 효과가 학습 데이터
규모와 synthetic 비율에 의존한다는 점을 보여준다. 가능한 원인은 생성
신호의 real-distribution 불일치, 낮은 생성 품질, 그리고 적은 real sample
상황에서 synthetic 신호가 학습을 과도하게 지배한 것이다.

이 결과를 제외하지 않고 ratio ablation의 기준으로 유지한다. 다음 실험은
real 3,420개를 고정한 채 `--augment-ratio 0.25`와 `0.5`를 측정하는 것이
적절하다.

Raw result file:
`/home/leehan/results/sensefi_har3_real10_aug100_random/result.json`

Status: **real 10% + synthetic 1:1 evaluation completed; performance decreased.**

#### 5. HAR-3 limited-real 10% + synthetic 25% — completed

Real train 3,420개를 동일하게 유지하고 synthetic CSI를 855개만 추가했다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --real-train-ratio 0.1 \
  --augment-data /mnt/ssd1/leehan/har3_generated_csi_sensefi_242.h5 \
  --augment-ratio 0.25 \
  --output-dir /home/leehan/results/sensefi_har3_real10_aug25_random \
  --model resnet18 \
  --split random-window \
  --epochs 50 \
  --seed 111
```

| Metric | Real 10% only | Real 10% + synthetic 25% | Change |
|---|---:|---:|---:|
| Accuracy | 89.7257% | **89.6985%** | **-0.0273%p** |
| Macro-F1 | 88.3879% | **88.2041%** | **-0.1837%p** |
| Macro-recall | 88.4240% | **88.3095%** | **-0.1145%p** |

Accuracy 차이는 -0.03%p로 매우 작아 실질적으로 baseline과 동일한 수준이다.
Synthetic 비율을 100%에서 25%로 줄이면 -3.17%p의 큰 성능 저하는
사라지지만, seed 111 단일 실행에서는 증강 이득도 확인되지 않았다. 작은
차이는 seed 변동 범위일 수 있으므로 반복 실행 전에는 통계적 차이로
해석하지 않는다.

Raw result file:
`/home/leehan/results/sensefi_har3_real10_aug25_random/result.json`

Status: **synthetic 25% evaluation completed; no measurable gain at seed 111.**

#### 6. HAR-3 limited-real 10% + synthetic 50% — completed

Real train 3,420개에 synthetic CSI 1,710개를 추가했다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --real-train-ratio 0.1 \
  --augment-data /mnt/ssd1/leehan/har3_generated_csi_sensefi_242.h5 \
  --augment-ratio 0.5 \
  --output-dir /home/leehan/results/sensefi_har3_real10_aug50_random \
  --model resnet18 \
  --split random-window \
  --epochs 50 \
  --seed 111
```

| Metric | Real 10% only | Real 10% + synthetic 50% | Change |
|---|---:|---:|---:|
| Accuracy | 89.7257% | **85.3050%** | **-4.4208%p** |
| Macro-F1 | 88.3879% | **83.1284%** | **-5.2595%p** |
| Macro-recall | 88.4240% | **83.3658%** | **-5.0582%p** |

Accuracy error rate는 10.2743%에서 14.6950%로 증가했으며, baseline 대비
오류가 **43.03% 상대 증가**했다. 이 결과는 synthetic 100%의 -3.17%p보다도
낮다. 따라서 synthetic 비율과 성능 사이에 단조로운 관계가 없으며, 단일
seed 학습 변동과 생성 데이터의 분포 불일치가 함께 영향을 줄 가능성이
있다.

Seed 111 기준 limited-real 실험의 최고 성능은 증강하지 않은 baseline
89.73%이며, 증강 조건 중에는 25%가 89.70%로 가장 높지만 baseline을
넘지 못했다. 최소한 baseline과 25% 조건을 여러 seed로 반복하기 전에는
25%를 최적 증강 비율로 주장할 수 없다.

Raw result file:
`/home/leehan/results/sensefi_har3_real10_aug50_random/result.json`

Status: **synthetic 50% evaluation completed; performance decreased.**

#### 7. HAR-3 limited-real 25% baseline — completed

전체 real train split 중 클래스별 25%를 사용한 baseline을 측정했다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --real-train-ratio 0.25 \
  --output-dir /home/leehan/results/sensefi_har3_real25_random \
  --model resnet18 \
  --split random-window \
  --epochs 50 \
  --seed 111
```

| Item | Value |
|---|---:|
| Available real train | 34,202 |
| Real train used | 8,550 (25%) |
| Validation / test | 7,329 / 7,329 |
| Accuracy | **95.1289%** |
| Macro-F1 | **94.3163%** |
| Macro-recall | **94.3715%** |

25%만 사용해도 클래스당 평균 약 427개 train window가 존재한다. 또한
`random-window` 분할에서는 같은 source trace에서 파생된 유사 window가
train과 test에 포함될 수 있으므로 participant, environment, recording
session 특성이 공유된다. 따라서 95.13%의 높은 값은 현재 in-domain
프로토콜에서 가능한 결과이며, unseen trace 또는 unseen environment
일반화를 의미하지 않는다.

Raw result file:
`/home/leehan/results/sensefi_har3_real25_random/result.json`

Status: **real 25% baseline completed.**

#### 8. HAR-3 source-file-grouped baseline — completed

동일한 HDF5 `source` 값에서 파생된 250-frame window를 하나의 split에만
배정하여 평가했다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --output-dir /home/leehan/results/sensefi_har3_source_trace \
  --model resnet18 \
  --split source-trace \
  --epochs 50 \
  --seed 111
```

| Item | Value |
|---|---:|
| Train / validation / test | 34,180 / 7,340 / 7,340 |
| Train / validation / test sources | 8,545 / 1,835 / 1,835 |
| Source overlap | **False** |
| Accuracy | **97.2616%** |
| Macro-F1 | **96.8744%** |
| Macro-recall | **96.8635%** |

Random-window baseline 97.7487%보다 accuracy가 0.4871%p 낮아졌지만 여전히
높다. 총 48,860개 window가 정확히 12,215개 source에서 파생되어 source당
평균 4개 window를 가진다. 이는 HDF5의 `source`가 물리적 수집 session
전체가 아니라 각 1,000-frame MAT 파일을 식별한다는 것을 보여준다.

따라서 이 실험은 동일 1,000-frame 파일에서 파생된 네 개의 250-frame
window가 split을 넘나드는 문제는 제거했지만, 동일 participant/environment/
recording session의 서로 다른 1,000-frame 파일 간 상관관계까지 제거한
진정한 session-held-out 평가는 아니다. 결과 명칭은 **source-file-grouped
in-domain baseline**으로 제한한다.

Raw result file:
`/home/leehan/results/sensefi_har3_source_trace/result.json`

Status: **source-file overlap removed; physical-session grouping requires
additional parent-trace metadata.**

#### 9. HAR-3 five-source-file-per-class baseline — completed

Source-file-grouped train fold에서 각 클래스당 1,000-frame MAT source file
5개만 선택했다. 각 source가 네 개의 250-frame window를 제공하므로 전체
real train은 400개다.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har3_official_csi_m1_242.h5 \
  --output-dir /home/leehan/results/sensefi_har3_5source \
  --model resnet18 \
  --split source-trace \
  --train-sources-per-class 5 \
  --epochs 50 \
  --seed 111
```

| Item | Value |
|---|---:|
| Train source files | 100 (5 × 20 classes) |
| Train / validation / test windows | 400 / 7,340 / 7,340 |
| Validation / test source files | 1,835 / 1,835 |
| Source overlap | **False** |
| Accuracy | **45.7357%** |
| Macro-F1 | **41.6869%** |
| Macro-recall | **41.9288%** |
| Random chance | 5.00% |

이 결과는 chance level보다 높지만 full source-file-grouped baseline
97.2616%보다 51.5259%p 낮다. 클래스당 train window가 약 20개뿐이므로
ResNet18이 학습하지 않은 source-file 변화에 일반화하기 어렵다는 것을
보여준다. 동시에 random-window real 10%의 89.73%와 비교하면 독립 source
수와 분할 방식이 단순 window 수보다 성능에 큰 영향을 준다는 점을
확인할 수 있다.

현재 `source`는 물리적 recording session이 아니라 1,000-frame MAT
source file이므로 이 결과를 real-world 또는 session-held-out 성능으로
표현하지 않고 **limited-source-file in-domain baseline**으로 보고한다.

Raw result file:
`/home/leehan/results/sensefi_har3_5source/result.json`

Status: **five-source-file-per-class baseline completed.**

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
