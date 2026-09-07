"""Create a source-disjoint BFA train/validation/test split.

The default HAR protocol uses one participant per fold so that windows from the
same PCAP/BFI capture never occur in more than one split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_participants(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("participants must be comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one participant is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-participants", type=parse_participants, default=(1,))
    parser.add_argument("--validation-participants", type=parse_participants, default=(2,))
    parser.add_argument("--test-participants", type=parse_participants, default=(3,))
    return parser.parse_args()


def canonical_source(value: object) -> str:
    name = Path(str(value)).name
    for suffix in (".pcapng", ".pcap", ".mat", ".npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def main() -> None:
    args = parse_args()
    folds = {
        "train": args.train_participants,
        "validation": args.validation_participants,
        "test": args.test_participants,
    }
    participant_sets = [set(values) for values in folds.values()]
    if any(participant_sets[i] & participant_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise SystemExit("train/validation/test participant sets must be disjoint")

    with np.load(args.npz, allow_pickle=False) as data:
        required = {"y", "participant", "source"}
        missing = required.difference(data.files)
        if missing:
            raise SystemExit(f"NPZ is missing fields: {sorted(missing)}")
        labels = np.asarray(data["y"], dtype=np.int64)
        participants = np.asarray(data["participant"], dtype=np.int64)
        sources = np.asarray([canonical_source(value) for value in data["source"]])

    available = set(int(value) for value in np.unique(participants))
    requested = set().union(*participant_sets)
    if requested != available:
        raise SystemExit(
            f"participant folds must cover the dataset exactly; available={sorted(available)}, "
            f"requested={sorted(requested)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices: dict[str, np.ndarray] = {}
    source_sets: dict[str, set[str]] = {}
    summary: dict[str, object] = {
        "protocol": "participant-source-disjoint-v1",
        "data": str(args.npz),
        "total_samples": int(len(labels)),
        "folds": {},
    }
    for name, fold_participants in folds.items():
        idx = np.flatnonzero(np.isin(participants, fold_participants)).astype(np.int64)
        if not len(idx):
            raise SystemExit(f"{name} split is empty")
        indices[name] = idx
        source_sets[name] = set(sources[idx].tolist())
        np.save(args.output_dir / f"{name}_indices.npy", idx)
        summary["folds"][name] = {
            "participants": list(fold_participants),
            "samples": int(len(idx)),
            "source_traces": int(len(source_sets[name])),
            "class_counts": np.bincount(labels[idx], minlength=20).astype(int).tolist(),
        }

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            raise SystemExit(f"source trace overlap between {left} and {right}: {sorted(overlap)[:5]}")

    combined = np.concatenate(list(indices.values()))
    if len(combined) != len(labels) or len(np.unique(combined)) != len(labels):
        raise SystemExit("generated folds do not form an exact partition of the dataset")

    (args.output_dir / "protocol.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
