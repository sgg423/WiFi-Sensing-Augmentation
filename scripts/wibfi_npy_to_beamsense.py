#!/usr/bin/env python3
"""Convert official Wi-BFI BFA NPY traces to BeamSense 10-frame NPZ data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


NAME = re.compile(
    r"^(?P<activity>[A-T])_(?P<day>\d+)_M(?P<monitor>\d+)_P(?P<participant>\d+)$",
    re.IGNORECASE,
)
ANGLE_NAMES = np.asarray(("phi_11", "phi_21", "psi_21", "psi_31"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Directory of official Wi-BFI BFA NPY files")
    parser.add_argument("output", type=Path, help="Output compressed BeamSense NPZ")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    if args.window_size <= 0 or args.stride <= 0:
        parser.error("--window-size and --stride must be positive")

    files = sorted(args.input.glob("*.npy"))
    if not files:
        raise SystemExit(f"No NPY files found in {args.input}")

    arrays = {key: [] for key in (
        "x", "y", "participant", "monitor", "repetition", "source", "window_start"
    )}
    total_frames = total_windows = 0
    for path in files:
        match = NAME.match(path.stem)
        if not match:
            raise ValueError(f"Unexpected BFA filename: {path.name}")
        bfa = np.load(path, mmap_mode="r")
        if bfa.ndim != 3 or bfa.shape[1:] != (234, 4):
            raise ValueError(f"{path.name}: expected [frames,234,4], got {bfa.shape}")
        if not np.issubdtype(bfa.dtype, np.integer):
            raise ValueError(f"{path.name}: expected integer quantized angles, got {bfa.dtype}")

        starts = np.arange(
            0, len(bfa) - args.window_size + 1, args.stride, dtype=np.int32
        )
        if not len(starts):
            print(f"SKIP {path.name}: only {len(bfa)} frames")
            continue
        windows = np.stack([
            np.asarray(bfa[start : start + args.window_size], dtype=np.uint16)
            for start in starts
        ])
        count = len(windows)
        activity = match.group("activity").upper()
        arrays["x"].append(windows)
        arrays["y"].append(np.full(count, ord(activity) - ord("A"), dtype=np.int16))
        arrays["participant"].append(
            np.full(count, int(match.group("participant")), dtype=np.int16)
        )
        arrays["monitor"].append(
            np.full(count, int(match.group("monitor")), dtype=np.int16)
        )
        arrays["repetition"].append(
            np.full(count, int(match.group("day")), dtype=np.int16)
        )
        arrays["source"].append(np.full(count, path.stem))
        arrays["window_start"].append(starts)
        total_frames += len(bfa)
        total_windows += count
        print(f"OK   {path.name}: frames={len(bfa)} windows={count}")

    if not arrays["x"]:
        raise SystemExit("No complete BeamSense windows were produced")
    output = {key: np.concatenate(value) for key, value in arrays.items()}
    output["activity_names"] = np.asarray(list("ABCDEFGHIJKLMNOPQRST"))
    output["angle_names"] = ANGLE_NAMES
    output["window_size"] = np.asarray(args.window_size, dtype=np.int16)
    output["stride"] = np.asarray(args.stride, dtype=np.int16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"Saved: {args.output}")
    print(f"x={output['x'].shape} dtype={output['x'].dtype}, y={output['y'].shape}")
    print(f"traces={len(files)} frames={total_frames} windows={total_windows}")


if __name__ == "__main__":
    main()
