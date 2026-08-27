"""Create conservative same-class BFA augmentation in circular angle space."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PERIOD = np.asarray([512.0, 512.0, 128.0, 128.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.alpha_min <= args.alpha_max < 0.5:
        raise SystemExit("Require 0 <= alpha-min <= alpha-max < 0.5")

    data = np.load(args.input, allow_pickle=False)
    x = data["x"]
    y = data["y"].astype(np.int64)
    if x.shape[1:] != (10, 234, 4):
        raise SystemExit(f"Expected x shape [N,10,234,4], got {x.shape}")

    rng = np.random.default_rng(args.seed)
    peer = np.empty(len(y), dtype=np.int64)
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        if len(members) == 1:
            peer[members] = members
            continue
        shuffled = rng.permutation(members)
        if np.any(shuffled == members):
            shuffled = np.roll(members, 1)
        peer[members] = shuffled

    alpha = rng.uniform(args.alpha_min, args.alpha_max, len(y)).astype(np.float32)
    output = np.empty_like(x, dtype=np.uint16)
    scale = (2.0 * np.pi / PERIOD).reshape(1, 1, 1, 4)

    for start in range(0, len(y), args.batch_size):
        end = min(start + args.batch_size, len(y))
        primary_angle = x[start:end].astype(np.float32) * scale
        peer_angle = x[peer[start:end]].astype(np.float32) * scale
        weight = alpha[start:end].reshape(-1, 1, 1, 1)

        mixed_real = (1.0 - weight) * np.cos(primary_angle) + weight * np.cos(peer_angle)
        mixed_imag = (1.0 - weight) * np.sin(primary_angle) + weight * np.sin(peer_angle)
        mixed_angle = np.mod(np.arctan2(mixed_imag, mixed_real), 2.0 * np.pi)
        indexes = np.rint(mixed_angle / scale).astype(np.int64)
        output[start:end] = np.mod(indexes, PERIOD.astype(np.int64)).astype(np.uint16)

    payload = {name: data[name] for name in data.files if name != "x"}
    payload.update(
        x=output,
        augmentation=np.asarray("same-class-circular-mixup"),
        mixup_peer=peer,
        mixup_alpha=alpha,
        augmentation_seed=np.asarray(args.seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Saved: {args.output}")
    print(f"x: {output.shape} {output.dtype}; y: {y.shape}")
    print(
        f"alpha: min={alpha.min():.4f} mean={alpha.mean():.4f} "
        f"max={alpha.max():.4f}"
    )
    print(f"identical samples: {np.all(output == x, axis=(1, 2, 3)).sum()}")


if __name__ == "__main__":
    main()
