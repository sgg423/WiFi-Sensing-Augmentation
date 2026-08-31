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
