# SenseFi Official CSI Experiment Log

## 날짜별 한눈에 보기

| 날짜 | 이날 수행한 작업 | 핵심 결과 | 상태 | 다음 작업 |
|---|---|---|---|---|
| 2026-08-11 | HAR-1 공식 complex CSI 구조 확인; `(1000,242)` CSI를 SenseFi 입력 `(1,250,242)`로 변환; SenseFi ResNet18 in-domain 및 3-fold cross-participant 평가 | In-domain **94.64%**; cross-participant accuracy **14.51±1.69%**; participant 14 실행은 테스트 샘플이 없어 무효 | 공식 CSI 변환, in-domain 및 cross-participant real-only baseline 완료 | 동일 fold에서 CSI 증강 후 성능 측정; cross-environment 분할 검토 |

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
