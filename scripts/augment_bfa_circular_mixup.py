"""Same-class BFA mixing: circular phi, bounded linear psi; train fold only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def mix_angles(primary, peer, alpha):
    """Use one weight per window, shared across time/subcarriers/angles."""
    a = np.asarray(primary, dtype=np.float64)
    b = np.asarray(peer, dtype=np.float64)
    weight = np.asarray(alpha, dtype=np.float64).reshape(-1, 1, 1, 1)
    # Phi quantization centers have a common half-bin offset, which cancels
    # when converting the circular interpolation back to index coordinates.
    scale = 2 * np.pi / 512
    phi_a, phi_b = a[..., :2] * scale, b[..., :2] * scale
    real = (1 - weight) * np.cos(phi_a) + weight * np.cos(phi_b)
    imag = (1 - weight) * np.sin(phi_a) + weight * np.sin(phi_b)
    phi = np.rint(np.mod(np.arctan2(imag, real), 2 * np.pi) / scale) % 512
    # Psi centers are affine in their index: interpolate without wraparound.
    psi = np.clip(np.rint((1 - weight) * a[..., 2:] + weight * b[..., 2:]), 0, 127)
    return np.concatenate((phi, psi), axis=-1).astype(np.uint16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--train-split-seed",
        type=int,
        required=True,
        help="restrict peers to the random-window training fold for this seed",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.alpha_min <= args.alpha_max < 0.5:
        raise SystemExit("Require 0 <= alpha-min <= alpha-max < 0.5")
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new filename")

    data = np.load(args.input, allow_pickle=False)
    x = data["x"]
    y = data["y"].astype(np.int64)
    if x.shape[1:] != (10, 234, 4):
        raise SystemExit(f"Expected x shape [N,10,234,4], got {x.shape}")
    if y.shape != (len(x),) or not len(x):
        raise SystemExit("Require nonempty x and one label per window")
    for channel, maximum in enumerate((511, 511, 127, 127)):
        values = x[..., channel]
        if (not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values > maximum) or np.any(values != np.floor(values))):
            raise SystemExit(f"Invalid quantized indices in angle {channel}")

    rng = np.random.default_rng(args.seed)
    eligible = np.ones(len(y), dtype=bool)
    if args.train_split_seed is not None:
        split_rng = np.random.default_rng(args.train_split_seed)
        eligible = split_rng.random(len(y)) < 0.70

    peer = np.arange(len(y), dtype=np.int64)
    for label in np.unique(y):
        members = np.flatnonzero((y == label) & eligible)
        if len(members) == 1:
            peer[members] = members
            continue
        shuffled = rng.permutation(members)
        if np.any(shuffled == members):
            shuffled = np.roll(members, 1)
        peer[members] = shuffled

    alpha = rng.uniform(args.alpha_min, args.alpha_max, len(y)).astype(np.float32)
    alpha[~eligible] = 0.0
    output = np.empty_like(x, dtype=np.uint16)

    for start in range(0, len(y), args.batch_size):
        end = min(start + args.batch_size, len(y))
        output[start:end] = mix_angles(x[start:end], x[peer[start:end]], alpha[start:end])
    output[~eligible] = x[~eligible]
    assert np.all(eligible[peer[eligible]])
    assert np.array_equal(y[peer], y)

    payload = {name: data[name] for name in data.files if name != "x"}
    payload.update(
        x=output,
        augmentation=np.asarray("same-class-phi-circular-psi-linear-v2"),
        augmentation_eligible=eligible,
        mixup_peer=peer,
        mixup_alpha=alpha,
        augmentation_seed=np.asarray(args.seed),
        train_split_seed=np.asarray(
            -1 if args.train_split_seed is None else args.train_split_seed
        ),
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
    print(f"eligible training samples: {eligible.sum()}")


if __name__ == "__main__":
    main()
