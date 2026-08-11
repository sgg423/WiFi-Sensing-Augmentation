# SenseFi Official CSI Experiment Log

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

The following example holds out participant 14.

```bash
cd /home/leehan/RF-Diffusion

python scripts/train_sensefi_har1.py \
  --data /home/leehan/datasets/har1_official_csi_242.h5 \
  --output-dir /home/leehan/results/sensefi_har1_official242_cross_user14 \
  --model resnet18 \
  --split participant \
  --test-participant 14 \
  --epochs 50 \
  --seed 111
```

### Results to record

| Held-out participant | Accuracy | Macro-F1 | Macro-recall | Result path |
|---:|---:|---:|---:|---|
| 14 | TBD | TBD | TBD | `/home/leehan/results/sensefi_har1_official242_cross_user14/result.json` |

When all participants have been evaluated, report both the individual results
and their mean and standard deviation. Do not combine this value with the
random-window in-domain result.

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
