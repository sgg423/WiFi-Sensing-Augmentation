# 2026-08-31: TimeVAE BFA comparison

Official code: https://github.com/abudesai/timeVAE
Pinned revision: 9914576bd44a6facbc94a594a82dd14724969a50.

This is **classwise official TimeVAE + BFA encoding/PCA**, not a conditional
TimeVAE implementation or an unchanged reproduction of the paper experiment.
Twenty independent class models sample latent normal priors. No source window
is fed to the decoder during generation. Metadata keys only allocate generated
samples to train slots for compatibility with the existing classifier trainer.

Preprocessing: quantized phi centers to cos/sin, psi to [-1,1], flatten each
frame's 234*6 features. Each class's train-only PCA reduces features to 32;
components are standardized using training standard deviations. This avoids
the official decoder's huge dense matrix at full 1404-dimensional input.
Generated PCA scores are inverted; phi is recovered with atan2 and quantized,
psi is clipped and quantized. PCA variance per class is logged. PCA may remove
discriminative information; it is part of this baseline, not a neutral reshape.

Model: official level+residual TimeVAE, filters [32,64], latent 8, reconstruction
weight 3, no polynomial/seasonality, fixed 100 epochs, batch 32. Save per-class
weights, preprocessors, history, and protocol. Smoke mode trains only class A
for two epochs and never creates a usable full-dataset augmentation file.

Only the BeamSense split111 training fold (28,529 windows) is accessed by
PCA and model training. Validation/test are not used by the generator.
The output has 28,529 generated samples, not the full 40,675 rows. BeamSense
matches all real train keys and then selects 2,852 at augmentation ratio .1.
Preserve this split; changing it requires regeneration. Class S has extremely
few real windows, so classwise training is unreliable for S. Do not claim
conditional fidelity from class labels assigned to outputs.

Dependencies: existing GPU TensorFlow environment + numpy, scikit-learn, joblib.
Do not install the upstream full requirements into the working sensing venv:
it pins unrelated older packages. No local TensorFlow training was available;
run the GPU smoke test before full training. Adapter tests cover exact
quantized roundtrip, range/finite checks, and existing split compatibility.

Compare against unchanged BeamSense at model/split/augmentation seed111 with
normalization none and balanced class weights. Use validation for method
selection; repeatedly inspected test results are exploratory. Independent final
holdout and multi-seed comparisons are required for confirmatory claims.

## PCA isolation diagnostic

Reported augmentation .1 result: Accuracy 0.8738901677079908,
macro F1 0.8821466340525091, macro recall 0.8845936866506385;
classifier/split/augmentation seeds 111. Versus historical fixed baseline
0.9409733640249918, change -6.7083 percentage points. This alone does not
identify PCA or generation as the cause.

`scripts/audit_timevae_pca_bfa.py` reloads the saved class PCA, never refits it,
and evaluates the same test indices through a frozen real-only classifier:
raw, angle encode/decode, PCA transform/inverse+angle decode. True labels route
the class PCA, so **PCA score is an oracle diagnostic, not deployable accuracy**.
All three are measured afresh from the same checkpoint. Historical logged
accuracy can differ because EarlyStopping's restored weights and the minimum
val-loss checkpoint do not necessarily coincide.

Automatic checkpoint discovery chooses the latest matching real-only
random-window baseline with explicit model/split seeds and matching input path;
prints all candidates and selection. `--baseline-dir` overrides selection.
Writes per-class confusion, metrics, indices and predictions. If PCA decreases
accuracy, preprocessing is a plausible contributor, not proof of the entire
augmentation decline. If PCA is stable, it still does not guarantee prior
samples are realistic. No retraining or modification of dataset files occurs.

Local adapter/checkpoint-selection tests pass; TensorFlow inference requires
GPU-server execution and is not claimed locally validated.
