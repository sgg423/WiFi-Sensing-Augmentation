#!/usr/bin/env python3
"""Convert HAR-1 RF MAT files into a disk-backed SenseFi input dataset."""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


OFFICIAL_NAME = re.compile(r"^user(?P<participant>\d+)_w(?P<label>\d+)$", re.IGNORECASE)


def metadata(path: Path, data):
    """Return zero-based activity label and available HAR-1 metadata."""
    cond = np.asarray(data.get("cond", [])).reshape(-1).astype(int)
    if cond.size >= 4 and 1 <= cond[0] <= 20:
        return cond[0] - 1, cond[3], cond[1], cond[2]

    match = OFFICIAL_NAME.match(path.stem)
    if match:
        label = int(match.group("label"))
        if 1 <= label <= 20:
            return label - 1, int(match.group("participant")), 0, 0
    return None


def files_and_counts(root: Path, window: int, stride: int, recursive: bool):
    records = []
    paths = root.rglob("user*.mat") if recursive else root.glob("user*.mat")
    width = None
    for path in sorted(paths):
        data = loadmat(path, variable_names=["feature", "cond", "source_filename"])
        if "feature" not in data:
            continue
        shape = np.asarray(data["feature"]).shape
        if len(shape) != 2:
            raise ValueError(f"{path}: expected feature [N,F], got {shape}")
        if width is None:
            width = shape[1]
        elif shape[1] != width:
            raise ValueError(f"{path}: feature width {shape[1]} differs from {width}")
        meta = metadata(path, data)
        if meta is None:
            print(f"SKIP {path.name}: activity/user metadata not found")
            continue
        count = max(0, 1 + (shape[0] - window) // stride)
        if count:
            records.append((path, count, meta))
    return records, width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--recursive", action="store_true",
                        help="search for user*.mat below subdirectories")
    args = parser.parse_args()

    records, width = files_and_counts(args.input_dir, args.window, args.stride, args.recursive)
    total = sum(record[1] for record in records)
    if not total:
        raise RuntimeError("No compatible HAR-1 MAT files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(args.output, "w") as out:
        x = out.create_dataset(
            "x", (total, 1, args.window, width), dtype=args.dtype,
            chunks=(1, 1, args.window, width), compression="lzf"
        )
        y = out.create_dataset("y", (total,), dtype="i2")
        participant = out.create_dataset("participant", (total,), dtype="i2")
        day = out.create_dataset("day", (total,), dtype="i2")
        monitor = out.create_dataset("monitor", (total,), dtype="i2")
        source = out.create_dataset("source", (total,), dtype=string_type)
        start_ds = out.create_dataset("window_start", (total,), dtype="i8")
        out.attrs.update(window=args.window, stride=args.stride, feature_width=width,
                         representation="abs(CSI)")

        cursor = 0
        for path, count, meta in records:
            data = loadmat(path, variable_names=["feature", "source_filename"])
            feature = np.asarray(data["feature"])
            source_name = str(np.asarray(data.get("source_filename", path.name)).squeeze())
            label, participant_id, day_id, monitor_id = meta
            for offset in range(0, count * args.stride, args.stride):
                x[cursor, 0] = np.abs(feature[offset:offset + args.window]).astype(args.dtype)
                y[cursor] = label
                participant[cursor] = participant_id
                day[cursor] = day_id
                monitor[cursor] = monitor_id
                source[cursor] = source_name
                start_ds[cursor] = offset
                cursor += 1
            print(f"OK {path.name}: windows={count}")
    print(f"Saved {args.output}: samples={total}, shape=({total},1,{args.window},{width})")


if __name__ == "__main__":
    main()
