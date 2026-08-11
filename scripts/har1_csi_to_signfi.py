#!/usr/bin/env python3
"""Convert official HAR-1 complex CSI MAT files into SignFi inputs."""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


OFFICIAL_NAME = re.compile(r"^user(?P<sample>\d+)_w(?P<label>\d+)$", re.IGNORECASE)


def metadata(path: Path, data):
    cond = np.asarray(data.get("cond", [])).reshape(-1).astype(int)
    if cond.size >= 4 and 1 <= cond[0] <= 20:
        return cond[0] - 1, cond[3], cond[1], cond[2]

    match = OFFICIAL_NAME.match(path.stem)
    if match and 1 <= int(match.group("label")) <= 20:
        # Official window files must retain cond to support domain splits. The
        # leading user#### token is a sample identifier, not a participant ID.
        return int(match.group("label")) - 1, 0, 0, 0
    return None


def sanitize_phase(csi):
    """Unwrap and remove a per-packet linear phase trend over subcarriers."""
    phase = np.unwrap(np.angle(csi), axis=1).astype(np.float32)
    width = phase.shape[1]
    index = np.arange(width, dtype=np.float32)
    centered = index - index.mean()
    denominator = np.sum(centered * centered)
    slopes = (phase @ centered) / max(float(denominator), 1e-8)
    intercepts = phase.mean(axis=1)
    return phase - slopes[:, None] * centered[None, :] - intercepts[:, None]


def discover(root, window, stride, recursive):
    paths = root.rglob("user*.mat") if recursive else root.glob("user*.mat")
    records, width = [], None
    for path in sorted(paths):
        data = loadmat(path, variable_names=["feature", "cond"])
        if "feature" not in data:
            continue
        feature = np.asarray(data["feature"])
        if feature.ndim != 2 or not np.iscomplexobj(feature):
            raise ValueError(f"{path}: expected complex feature [N,F], got {feature.shape} {feature.dtype}")
        if width is None:
            width = feature.shape[1]
        elif feature.shape[1] != width:
            raise ValueError(f"{path}: feature width {feature.shape[1]} differs from {width}")
        meta = metadata(path, data)
        if meta is None:
            print(f"SKIP {path.name}: activity metadata not found")
            continue
        count = max(0, 1 + (feature.shape[0] - window) // stride)
        if count:
            records.append((path, count, meta))
    return records, width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--phase", choices=["raw", "sanitized"], default="sanitized")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    records, width = discover(args.input_dir, args.window, args.stride, args.recursive)
    total = sum(record[1] for record in records)
    if not total:
        raise RuntimeError("No compatible official HAR-1 CSI MAT files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(args.output, "w") as out:
        x = out.create_dataset(
            "x", (total, 2, args.window, width), dtype=args.dtype,
            chunks=(1, 2, args.window, width), compression="lzf"
        )
        y = out.create_dataset("y", (total,), dtype="i2")
        participant = out.create_dataset("participant", (total,), dtype="i2")
        day = out.create_dataset("day", (total,), dtype="i2")
        monitor = out.create_dataset("monitor", (total,), dtype="i2")
        source = out.create_dataset("source", (total,), dtype=string_type)
        start_ds = out.create_dataset("window_start", (total,), dtype="i8")
        out.attrs.update(
            window=args.window, stride=args.stride, feature_width=width,
            representation=f"SignFi amplitude + {args.phase} phase",
            signfi_tensor=f"amplitude/phase concatenated to [1,{args.window},{2 * width}] at load time"
        )

        cursor = 0
        for path, count, meta in records:
            data = loadmat(path, variable_names=["feature", "source_filename"])
            feature = np.asarray(data["feature"])
            amplitude = np.abs(feature).astype(np.float32)
            phase = (sanitize_phase(feature) if args.phase == "sanitized"
                     else np.angle(feature).astype(np.float32))
            source_name = str(np.asarray(data.get("source_filename", path.name)).squeeze())
            label, participant_id, day_id, monitor_id = meta
            for offset in range(0, count * args.stride, args.stride):
                stop = offset + args.window
                x[cursor, 0] = amplitude[offset:stop].astype(args.dtype)
                x[cursor, 1] = phase[offset:stop].astype(args.dtype)
                y[cursor] = label
                participant[cursor] = participant_id
                day[cursor] = day_id
                monitor[cursor] = monitor_id
                source[cursor] = source_name
                start_ds[cursor] = offset
                cursor += 1
            print(f"OK {path.name}: windows={count}")

    print(f"Saved {args.output}: samples={total}, stored=({total},2,{args.window},{width}), "
          f"SignFi input=({total},1,{args.window},{2 * width})")


if __name__ == "__main__":
    main()
