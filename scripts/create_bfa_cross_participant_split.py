"""Create a cross-participant BFA split with chronological validation.

Training and validation windows come from the designated training participant.
For every source trace, its chronological tail is held out for validation. The
test participant is completely source-disjoint. Other participants are excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-participant", type=int, required=True)
    parser.add_argument("--test-participant", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    return parser.parse_args()


def canonical_source(value: object) -> str:
    name = Path(str(value)).name
    for suffix in (".pcapng", ".pcap", ".mat", ".npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def main() -> None:
    args = parse_args()
    if args.train_participant == args.test_participant:
        raise SystemExit("train and test participants must differ")
    if not 0 < args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be between 0 and 0.5")

    with np.load(args.npz, allow_pickle=False) as data:
        required = {"y", "participant", "source", "window_start"}
        missing = required.difference(data.files)
        if missing:
            raise SystemExit(f"NPZ is missing fields: {sorted(missing)}")
        labels = np.asarray(data["y"], dtype=np.int64)
        participants = np.asarray(data["participant"], dtype=np.int64)
        sources = np.asarray([canonical_source(value) for value in data["source"]])
        starts = np.asarray(data["window_start"], dtype=np.int64)

    train_candidates = np.flatnonzero(participants == args.train_participant)
    test = np.flatnonzero(participants == args.test_participant).astype(np.int64)
    if not len(train_candidates) or not len(test):
        raise SystemExit("selected train or test participant has no samples")

    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for source in np.unique(sources[train_candidates]):
        indexes = train_candidates[sources[train_candidates] == source]
        indexes = indexes[np.argsort(starts[indexes], kind="stable")]
        validation_count = max(1, int(np.floor(len(indexes) * args.validation_fraction)))
        if validation_count >= len(indexes):
            raise SystemExit(f"source {source} is too short for a validation tail")
        train_parts.append(indexes[:-validation_count])
        validation_parts.append(indexes[-validation_count:])

    train = np.concatenate(train_parts).astype(np.int64)
    validation = np.concatenate(validation_parts).astype(np.int64)
    train_sources = set(sources[train].tolist())
    test_sources = set(sources[test].tolist())
    overlap = train_sources & test_sources
    if overlap:
        raise SystemExit(f"train/test source overlap: {sorted(overlap)[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, indexes in (("train", train), ("validation", validation), ("test", test)):
        np.save(args.output_dir / f"{name}_indices.npy", indexes)

    used = np.concatenate((train, validation, test))
    protocol = {
        "protocol": "cross-participant-chronological-validation-v1",
        "data": str(args.npz),
        "train_participant": args.train_participant,
        "test_participant": args.test_participant,
        "validation_fraction": args.validation_fraction,
        "train_test_source_overlap": False,
        "excluded_samples": int(len(labels) - len(used)),
        "folds": {},
    }
    for name, indexes in (("train", train), ("validation", validation), ("test", test)):
        protocol["folds"][name] = {
            "samples": int(len(indexes)),
            "source_traces": int(len(np.unique(sources[indexes]))),
            "class_counts": np.bincount(labels[indexes], minlength=20).astype(int).tolist(),
        }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
