# HAR-3 CSI fixed-seed re-evaluation (2026-08-31)

## Scope

Compare SenseFi ResNet18 on all available **real training data** against that
same training fold plus 1:1 synthetic CSI. The real validation/test folds are
not used as classifier training samples. "100% real" means the full training
fold, not the entire dataset including validation/test.

- Real HDF5: `/home/leehan/datasets/har3_official_csi_m1_242.h5`
- Synthetic HDF5: `/mnt/ssd1/leehan/har3_generated_csi_sensefi_242.h5`
- Expected input tensor per sample: `(1, 250, 242)` (inspect actual files).
- Split seed: 111; synthetic selection seed: 111.
- Model seeds: 42, 111, 2026, 3407, 7777.
- Full real training fold; synthetic/real train ratio: 1.0.
- Expected real train/validation/test: 34,202 / 7,329 / 7,329.
- Expected paired synthetic train: 34,202.
- Normalization statistics: real training fold only.
- Classifier training: original ResNet18, batch size 64, Adam lr 0.001,
  maximum 50 epochs, patience 8, best validation accuracy checkpoint.

## Changes from historical runs

`--split-seed` controls both split creation and any optional real-data limits.
`--augment-seed` controls synthetic subsampling. `--seed` controls classifier
initialization and training loader order. DataLoader generators are separated
from model initialization and validation/test iteration.

`--augment-match train-window` matches source basename, within-source window
offset, and activity label. It rejects missing or duplicate metadata and only
admits synthetic counterparts of the real training windows. It cannot detect
incorrect provenance written into those fields. Historical `all` pool selection
remains available explicitly/default for compatibility, with a warning.

At 1:1 with exactly one paired synthetic per train window, all eligible pairs
are used: changing augmentation seed does not change that selection.

Historical 97.75% baseline and 98.39% augmented accuracy are references, not
baselines to reuse in this protocol. Re-run both conditions for all model seeds.
This is a stricter re-evaluation, not an exact replication of all-pool sampling.

## Audit artifacts

Every run, including `--dry-run`, saves:

- `selection_indices.npz`: real train/validation/test and synthetic indices.
- `protocol.json`: paths, shapes, counts, seeds, SHA-256 of index arrays and code.

For a fixed pair of input files, real fold hashes must agree between baseline
and augmentation and across all model seeds. Synthetic hashes must agree across
augmented runs. Do not modify HDF5 files during the experiment. A final training
run also saves `result.json`, `history.csv`, and `best.pt`.

## Provenance limitation

Matching synthetic training counterparts does **not** undo test-data exposure
inside the generator. Confirm which source files were used to train RF-Diffusion
and any guidance classifier. If validation/test sources were used there, label
these runs exploratory; a clean held-out claim requires generator/guidance
training restricted to the appropriate training fold.

Random-window folds may contain correlated windows from the same capture. They
measure within-dataset performance, not unseen-capture/environment generalization.

## Validation performed locally

`tests/test_sensefi_protocol.py`: six data-protocol tests using synthetic HDF5s,
covering fixed folds/subsets across model seeds, independent augmentation and
split seeds, matched train-only selection, missing metadata/matches, and duplicate
keys. These tests execute extracted data/dry-run functions without PyTorch; they
do not test GPU optimization or claim numerical training reproducibility.
