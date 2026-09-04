"""Select one high-quality BFA diffusion candidate for every real training anchor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


MOD = np.asarray([512, 512, 128, 128], dtype=np.int32)
CLASS_NAMES = np.asarray(list("ABCDEFGHIJKLMNOPQRST"))


def temporal_features(x: np.ndarray) -> np.ndarray:
    """Mean absolute circular frame delta for each of the four BFA angles."""
    delta = np.diff(x.astype(np.int32), axis=1)
    delta = (delta + MOD // 2) % MOD - MOD // 2
    return np.mean(np.abs(delta), axis=(1, 2), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_npz", type=Path)
    parser.add_argument("candidate_npz", type=Path)
    parser.add_argument("teacher_model", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--realism-weight", type=float, default=0.15)
    parser.add_argument("--incorrect-penalty", type=float, default=5.0)
    args = parser.parse_args()

    import tensorflow as tf

    real = np.load(args.real_npz, allow_pickle=False)
    candidates = np.load(args.candidate_npz, allow_pickle=False)
    cx = candidates["x"]
    cy = candidates["y"].astype(np.int64)
    anchors = candidates["allocation_real_index"].astype(np.int64)
    if cx.shape[1:] != (10, 234, 4):
        raise SystemExit(f"Expected candidate shape (N,10,234,4), got {cx.shape}")
    if not (len(cx) == len(cy) == len(anchors)):
        raise SystemExit("Candidate x/y/allocation_real_index length mismatch")
    if np.any(cy != real["y"][anchors]):
        raise SystemExit("Candidate labels do not match their real anchors")

    unique_anchors, counts = np.unique(anchors, return_counts=True)
    if len(np.unique(counts)) != 1:
        raise SystemExit(f"Every anchor must have the same candidate count, got {np.unique(counts)}")
    candidates_per_anchor = int(counts[0])
    expected = np.repeat(unique_anchors, candidates_per_anchor)
    if not np.array_equal(anchors, expected):
        raise SystemExit("Candidates must be grouped contiguously by allocation_real_index")

    class CandidateSequence(tf.keras.utils.Sequence):
        def __len__(self):
            return int(np.ceil(len(cx) / args.batch_size))

        def __getitem__(self, batch):
            start = batch * args.batch_size
            stop = min(start + args.batch_size, len(cx))
            return cx[start:stop].astype(np.float32)

    teacher = tf.keras.models.load_model(args.teacher_model)
    probabilities = teacher.predict(CandidateSequence(), verbose=1)
    target_confidence = probabilities[np.arange(len(cy)), cy]
    prediction = np.argmax(probabilities, axis=1)

    real_y = real["y"].astype(np.int64)
    real_temporal = temporal_features(real["x"][unique_anchors])
    class_mean = np.empty((20, 4), dtype=np.float64)
    class_std = np.empty((20, 4), dtype=np.float64)
    for label in range(20):
        values = real_temporal[real_y[unique_anchors] == label]
        if not len(values):
            raise SystemExit(f"No real training anchor for class {label}")
        class_mean[label] = values.mean(axis=0)
        class_std[label] = np.maximum(values.std(axis=0), 1e-3)

    candidate_temporal = temporal_features(cx)
    temporal_z = np.mean(
        np.abs(candidate_temporal - class_mean[cy]) / class_std[cy], axis=1
    )
    score = (
        np.log(np.maximum(target_confidence, 1e-8))
        - args.realism_weight * temporal_z
        - args.incorrect_penalty * (prediction != cy)
    )
    grouped_score = score.reshape(len(unique_anchors), candidates_per_anchor)
    selected = (
        np.arange(len(unique_anchors)) * candidates_per_anchor
        + np.argmax(grouped_score, axis=1)
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "x": cx[selected],
        "y": cy[selected],
        "allocation_real_index": anchors[selected],
        "augmentation_eligible": np.ones(len(selected), dtype=bool),
        "selection_score": score[selected].astype(np.float32),
        "teacher_target_confidence": target_confidence[selected].astype(np.float32),
        "teacher_prediction": prediction[selected].astype(np.int16),
        "temporal_z": temporal_z[selected].astype(np.float32),
    }
    for key in ("train_split_seed", "augmentation"):
        if key in candidates.files:
            output[key] = candidates[key]
    for key in ("source", "window_start", "participant", "candidate_rank"):
        if key in candidates.files:
            output[key] = candidates[key][selected]
    np.savez_compressed(args.output_npz, **output)

    selected_prediction = prediction[selected]
    report = {
        "real_npz": str(args.real_npz),
        "candidate_npz": str(args.candidate_npz),
        "teacher_model": str(args.teacher_model),
        "output_npz": str(args.output_npz),
        "anchors": len(unique_anchors),
        "candidates_per_anchor": candidates_per_anchor,
        "candidate_teacher_accuracy": float(np.mean(prediction == cy)),
        "selected_teacher_accuracy": float(np.mean(selected_prediction == cy[selected])),
        "selected_mean_target_confidence": float(np.mean(target_confidence[selected])),
        "selected_mean_temporal_z": float(np.mean(temporal_z[selected])),
        "selected_class_counts": {
            name: int(count)
            for name, count in zip(CLASS_NAMES, np.bincount(cy[selected], minlength=20))
        },
        "realism_weight": args.realism_weight,
        "incorrect_penalty": args.incorrect_penalty,
    }
    report_path = args.output_npz.with_suffix(".selection.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
