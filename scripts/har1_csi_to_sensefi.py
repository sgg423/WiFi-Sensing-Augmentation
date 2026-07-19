#!/usr/bin/env python3
"""Convert HAR-1 RF MAT files into a disk-backed SenseFi input dataset."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def files_and_counts(root: Path, window: int, stride: int):
    records = []
    for path in sorted(root.glob("user*.mat")):
        data = loadmat(path, variable_names=["feature", "cond", "source_filename"])
        shape = np.asarray(data["feature"]).shape
        cond = np.asarray(data["cond"]).reshape(-1).astype(int)
        if len(shape) != 2 or shape[1] != 90:
            raise ValueError(f"{path}: expected feature [N,90], got {shape}")
        if cond.size < 4 or not 1 <= cond[0] <= 20:
            continue
        count = max(0, 1 + (shape[0] - window) // stride)
        if count:
            records.append((path, count, cond[:4]))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    records = files_and_counts(args.input_dir, args.window, args.stride)
    total = sum(record[1] for record in records)
    if not total:
        raise RuntimeError("No compatible HAR-1 MAT files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(args.output, "w") as out:
        x = out.create_dataset(
            "x", (total, 1, args.window, 90), dtype=args.dtype,
            chunks=(1, 1, args.window, 90), compression="lzf"
        )
        y = out.create_dataset("y", (total,), dtype="i2")
        participant = out.create_dataset("participant", (total,), dtype="i2")
        day = out.create_dataset("day", (total,), dtype="i2")
        monitor = out.create_dataset("monitor", (total,), dtype="i2")
        source = out.create_dataset("source", (total,), dtype=string_type)
        start_ds = out.create_dataset("window_start", (total,), dtype="i8")
        out.attrs.update(window=args.window, stride=args.stride, representation="abs(CSI)")

        cursor = 0
        for path, count, cond in records:
            data = loadmat(path, variable_names=["feature", "source_filename"])
            feature = np.asarray(data["feature"])
            source_name = str(np.asarray(data.get("source_filename", path.name)).squeeze())
            for offset in range(0, count * args.stride, args.stride):
                x[cursor, 0] = np.abs(feature[offset:offset + args.window]).astype(args.dtype)
                y[cursor] = cond[0] - 1
                day[cursor], monitor[cursor], participant[cursor] = cond[1:4]
                source[cursor] = source_name
                start_ds[cursor] = offset
                cursor += 1
            print(f"OK {path.name}: windows={count}")
    print(f"Saved {args.output}: samples={total}, shape=({total},1,{args.window},90)")


if __name__ == "__main__":
    main()
