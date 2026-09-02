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

## 2026-09-02 — RF-Diffusion native `t` sweep

Four full generated pools were converted from complex V to BFA and evaluated
with the same frozen BeamSense checkpoint. Each pool contains 40,675 windows.
The meaning of `t` must be confirmed from the generator configuration; this log
does not assume whether it denotes a diffusion timestep, starting noise level,
or another parameter.

| Generated pool | Label agreement | Macro F1 | Macro recall | Dominant predictions |
|---|---:|---:|---:|---|
| `M1_w10_native_t25_generated` | 17.6546% | 12.4817% | 17.4166% | D 28.34%, G 22.55%, A 11.86% |
| `M1_w10_native_t50_generated` | 12.2311% | 7.1820% | 12.4798% | G 48.32%, A 20.96% |
| `M1_w10_native_t75_generated` | 5.9988% | 2.3389% | 6.7770% | A 51.72%, G 37.29% |
| `M1_w10_native_t99_generated` | 4.8310% | 1.3899% | 5.7202% | A 76.15%, G 21.13% |

All three classification metrics decrease as `t` increases. Prediction mass
also progressively collapses toward A/G: their combined fraction is 34.41%,
69.28%, 89.01%, and 97.28% for t25, t50, t75, and t99 respectively. t25 is the
best of this sweep but remains far below the frozen classifier's performance on
real BFA and is highly imbalanced. None of these pools is promoted to the main
downstream augmentation comparison based on this diagnostic.

This sweep establishes an empirical association, not a mechanism. Required
generator-side checks are: exact definition and application of `t`; whether
conditioning strength changes with `t`; identical seeds/conditions/checkpoints
across the sweep; reverse-process completion; complex-V normalization and
postprocessing; and metadata/label alignment. Generated-label agreement is not
itself downstream sensing accuracy, and the frozen classifier is not an
independent physical-fidelity metric.

### TimeVAE-BFA 1:1 paired multi-seed comparison

Class-wise TimeVAE generated 28,529 BFA windows from the train-only split
(generator seed 42, split seed 111). BeamSense was trained with the real 28,529
and all generated 28,529 windows. The generated pool and its ordered index hash
were identical for every model seed (`03eae5a47bddca82821a871442f1d70bede860b3db3fac742b2661463501aa1c`).

| Model seed | Real-only accuracy | TimeVAE 1:1 accuracy | Paired gain |
|---:|---:|---:|---:|
| 42 | 93.5054% | 94.9194% | +1.4140%p |
| 111 | 93.7849% | 91.5653% | -2.2197%p |
| 2026 | 91.3515% | 93.9658% | +2.6143%p |
| 3407 | 85.1529% | 91.7955% | +6.6426%p |
| 7777 | 94.3275% | 93.5712% | -0.7563%p |
| **Mean ± sample SD** | **91.6245 ± 3.7904%** | **93.1634 ± 1.4421%** | **+1.5390 ± 3.4137%p** |

Macro-F1 changed from 91.7092 ± 4.2450% to 93.5048 ± 1.3681%, with
paired gain +1.7956 ± 3.7883%p. Macro recall changed from 92.0131 ± 3.5758%
to 93.4707 ± 1.4404%, with paired gain +1.4576 ± 3.2180%p.

Accuracy improved for three of five model seeds and decreased for two. The
mean accuracy increased and between-seed sample SD decreased by 2.3482%p, but
the paired gain SD exceeds its mean. With only five model seeds, this is
promising evidence of average improvement/stabilization, not a stable or
statistically established universal gain. TimeVAE is an adapted class-wise
baseline (BFA angle encoding + train-only PCA), not the proposed contribution.
