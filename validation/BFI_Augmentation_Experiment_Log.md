# BFI augmentation experiment log

## 2026-09-01 — BeamSense real-only multi-seed baseline

### Fixed evaluation protocol

- Dataset: HAR-1 Kitchen, M1 BFA.
- Input: `(10, 234, 4)` quantized BFA, `uint16`.
- Split: random window, fixed `split_seed=111`.
- Fold sizes: train 28,529 / validation 6,064 / test 6,082.
- Model: BeamSense CNN; epochs 100; balanced class weight; no normalization.
- Only the model initialization/training seed changes.
- Script revision: `b231e5b` (`train_beamsense_har1.py`).

| Model seed | Accuracy | Macro F1 | Macro recall |
|---:|---:|---:|---:|
| 42 | 93.5054% | 93.8353% | 93.8909% |
| 111 | 93.7849% | 93.9882% | 93.8800% |
| 2026 | 91.3515% | 91.8139% | 91.8570% |
| 3407 | 85.1529% | 84.3461% | 85.8789% |
| 7777 | 94.3275% | 94.5625% | 94.5590% |
| **Mean ± sample SD** | **91.6245 ± 3.7904%** | **91.7092 ± 4.2450%** | **92.0131 ± 3.5758%** |

Observed accuracy range: 85.1529–94.3275%. The large model-seed variation
means a single-run sub-percentage-point gain cannot be treated as stable
augmentation evidence. Seed 3407 is retained; it is not excluded as an outlier
without a predefined exclusion rule or demonstrated execution failure.

Final augmentation comparisons must use the same five model seeds and report
paired per-seed gain (`augmented - real-only`), mean gain, and sample standard
deviation. Split seed remains 111. For a fixed generated pool, augmentation seed
remains 111. Baseline and augmented runs use the same train/validation/test fold.

### RF-Diffusion balanced 100-sample generation diagnostics

Generated complex V samples were converted to BFA with the validated
`vmatrix_to_bfa.py` path and evaluated with one frozen BeamSense checkpoint.
These are generated-label agreement diagnostics, not downstream real-test
augmentation accuracies. Each set contains only 100 samples; a one-sample
difference is one percentage point.

| RF-Diffusion variant | Samples | Label agreement | Prediction-distribution observation |
|---|---:|---:|---|
| `test_tail_full_balanced_100` | 100 | 58% | D 22%, G 15%; 8 classes never predicted |
| `test_feature_full_balanced_100` | 100 | 60% | A/B/D 51% combined; 6 classes never predicted |
| `test_delta_feature_full_balanced_100` | 100 | 5% | N 95%, A 5%; all other classes 0% |

Feature-full is two correct samples above tail-full, which is insufficient to
claim superiority at this sample size. Delta-feature shows severe prediction
collapse and should not proceed to downstream augmentation without checking
delta reconstruction, axis/direction, feature scale, and label/condition
alignment. Real BFA reaches high accuracy at 10 frames, so the window length
alone is not established as the cause of generated-signal collapse.

The 100-sample files match only a small subset of the real training fold. The
partial-augmentation option filters generated rows to matching training keys and
does not use validation/test counterparts. Approximately 70 eligible synthetic
samples would be less than 0.3% of the 28,529 real training windows; any resulting
accuracy change is expected to be difficult to distinguish from training
variation. A full class-balanced generated pool is needed for the paired
five-seed downstream comparison.

### Current status

- Multi-seed real-only baseline: completed.
- Balanced 100-sample generated-label diagnostics: completed for tail-full,
  feature-full, and delta-feature-full.
- RF-Diffusion BFA downstream augmentation: pending a sufficiently large,
  class-balanced generated pool.
- Results are exploratory because the same test split has been inspected during
  method development. A final untouched split/environment is required after the
  proposed method is fixed.
