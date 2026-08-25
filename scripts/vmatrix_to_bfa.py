#!/usr/bin/env python3
"""Recover 3x1 MU-MIMO BFA indices from Wi-BFI complex V matrices.

The CSI-BFI-HAR captures use four quantized angles in this order:
phi_11, phi_21, psi_21, psi_31, with 9/9/7/7 bits.  This script can
validate the inverse against a paired Wi-BFI V/BFA trace or convert a
directory of RF-Diffusion MAT samples into a BeamSense-compatible NPZ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ANGLE_NAMES = np.asarray(("phi_11", "phi_21", "psi_21", "psi_31"))
ANGLE_SCALE = np.asarray((511, 511, 127, 127), dtype=np.uint16)


def vmatrix_to_bfa(vmatrix: np.ndarray) -> np.ndarray:
    """Convert [..., 3, 1] complex V vectors to [..., 4] BFA indices."""
    value = np.asarray(vmatrix)
    if value.shape[-2:] != (3, 1):
        raise ValueError(f"expected V shape [...,3,1], got {value.shape}")
    if not np.iscomplexobj(value):
        raise ValueError("V matrix must be complex")
    if not np.isfinite(value).all():
        raise ValueError("V matrix contains NaN or Inf")

    vector = value[..., 0].astype(np.complex128, copy=False)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float64).eps):
        raise ValueError("V matrix contains a zero-norm vector")
    vector = vector / norm

    # IEEE 802.11 3x1 parameterization fixes the final component to a
    # non-negative real value. Remove the otherwise irrelevant global phase.
    vector = vector * np.exp(-1j * np.angle(vector[..., 2:3]))

    phi_11 = np.mod(np.angle(vector[..., 0]), 2 * np.pi)
    phi_21 = np.mod(np.angle(vector[..., 1]), 2 * np.pi)
    psi_21 = np.arctan2(np.abs(vector[..., 1]), np.abs(vector[..., 0]))
    psi_31 = np.arcsin(np.clip(np.abs(vector[..., 2]), 0.0, 1.0))

    # Wi-BFI reconstructs angle centers as:
    # phi = pi * (1/2^9 + q/2^8), psi = pi * (1/2^9 + q/2^8).
    def quantize_phi(angle: np.ndarray) -> np.ndarray:
        index = np.rint(angle * (256.0 / np.pi) - 0.5).astype(np.int64)
        return np.mod(index, 512).astype(np.uint16)

    def quantize_psi(angle: np.ndarray) -> np.ndarray:
        index = np.rint(angle * (256.0 / np.pi) - 0.5).astype(np.int64)
        return np.clip(index, 0, 127).astype(np.uint16)

    return np.stack(
        (
            quantize_phi(phi_11),
            quantize_phi(phi_21),
            quantize_psi(psi_21),
            quantize_psi(psi_31),
        ),
        axis=-1,
    )


def bfa_to_vmatrix(bfa: np.ndarray) -> np.ndarray:
    """Reconstruct [..., 3, 1] V vectors using Wi-BFI's 3x1 formula."""
    angle = np.asarray(bfa)
    if angle.shape[-1] != 4:
        raise ValueError(f"expected BFA shape [...,4], got {angle.shape}")
    phi_11 = np.pi * (1 / 512 + angle[..., 0] / 256)
    phi_21 = np.pi * (1 / 512 + angle[..., 1] / 256)
    psi_21 = np.pi * (1 / 512 + angle[..., 2] / 256)
    psi_31 = np.pi * (1 / 512 + angle[..., 3] / 256)

    cos_31 = np.cos(psi_31)
    vector = np.stack(
        (
            np.exp(1j * phi_11) * np.cos(psi_21) * cos_31,
            np.exp(1j * phi_21) * np.sin(psi_21) * cos_31,
            np.sin(psi_31).astype(np.complex128),
        ),
        axis=-1,
    )
    return vector[..., None]


def string_value(value: np.ndarray, fallback: str) -> str:
    if value is None:
        return fallback
    squeezed = np.asarray(value).squeeze()
    return str(squeezed) if squeezed.size else fallback


def validate(args: argparse.Namespace) -> None:
    v = np.load(args.vmatrix, mmap_mode="r")
    expected = np.load(args.bfa, mmap_mode="r")
    if args.max_frames is not None:
        v = v[: args.max_frames]
        expected = expected[: args.max_frames]
    recovered = vmatrix_to_bfa(v)
    if recovered.shape != expected.shape:
        raise ValueError(
            f"recovered shape {recovered.shape} differs from BFA {expected.shape}"
        )

    difference = recovered.astype(np.int32) - expected.astype(np.int32)
    exact_per_angle = np.mean(difference == 0, axis=tuple(range(difference.ndim - 1)))
    within_one = np.mean(
        np.abs(difference) <= 1, axis=tuple(range(difference.ndim - 1))
    )
    all_exact = np.mean(np.all(difference == 0, axis=-1))

    reconstructed = bfa_to_vmatrix(recovered)[..., 0]
    original = np.asarray(v)[..., 0]
    original = original / np.linalg.norm(original, axis=-1, keepdims=True)
    inner = np.sum(np.conj(original) * reconstructed, axis=-1)
    cosine = np.abs(inner)

    print("frames:", len(v))
    print("exact per angle:", dict(zip(ANGLE_NAMES.tolist(), exact_per_angle.tolist())))
    print("within +/-1 per angle:", dict(zip(ANGLE_NAMES.tolist(), within_one.tolist())))
    print("all-angle exact rate:", float(all_exact))
    print("phase-invariant cosine mean:", float(np.mean(cosine)))
    print("phase-invariant cosine minimum:", float(np.min(cosine)))


def convert_mat(args: argparse.Namespace) -> None:
    from scipy.io import loadmat

    files = sorted(args.input.rglob("*.mat"))
    if not files:
        raise SystemExit(f"No MAT files found in {args.input}")

    sample = loadmat(files[0], variable_names=["feature"])["feature"]
    if sample.shape != (10, 234, 3, 1):
        raise ValueError(f"expected feature (10,234,3,1), got {sample.shape}")

    count = len(files)
    x = np.empty((count, 10, 234, 4), dtype=np.uint16)
    y = np.empty(count, dtype=np.int16)
    participant = np.empty(count, dtype=np.int16)
    monitor = np.empty(count, dtype=np.int16)
    repetition = np.empty(count, dtype=np.int16)
    source = np.empty(count, dtype="U64")
    window_start = np.empty(count, dtype=np.int32)

    norm_error_sum = 0.0
    norm_error_max = 0.0
    for index, path in enumerate(files):
        data = loadmat(
            path,
            variable_names=[
                "feature", "cond", "parent_trace", "source_filename",
                "window_start",
            ],
        )
        metadata = data
        if args.metadata_dir is not None:
            relative = path.relative_to(args.input)
            metadata_path = args.metadata_dir / relative
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"{path}: paired metadata MAT not found at {metadata_path}"
                )
            metadata = loadmat(
                metadata_path,
                variable_names=[
                    "cond", "parent_trace", "source_filename", "window_start"
                ],
            )
        feature = np.asarray(data["feature"])
        cond = np.asarray(data["cond"]).reshape(-1).astype(int)
        if feature.shape != (10, 234, 3, 1):
            raise ValueError(f"{path}: unexpected feature shape {feature.shape}")
        if len(cond) < 4 or not 1 <= cond[0] <= 20:
            raise ValueError(f"{path}: invalid cond {cond}")
        metadata_cond = np.asarray(metadata.get("cond", data["cond"])).reshape(-1).astype(int)
        if len(metadata_cond) >= 4 and not np.array_equal(cond[:4], metadata_cond[:4]):
            raise ValueError(
                f"{path}: generated cond {cond[:4]} differs from paired real "
                f"cond {metadata_cond[:4]}"
            )

        norms = np.linalg.norm(feature[..., 0], axis=-1)
        errors = np.abs(norms - 1.0)
        norm_error_sum += float(np.sum(errors))
        norm_error_max = max(norm_error_max, float(np.max(errors)))

        x[index] = vmatrix_to_bfa(feature)
        y[index] = cond[0] - 1
        repetition[index] = cond[1]
        monitor[index] = cond[2]
        participant[index] = cond[3]
        source[index] = string_value(
            metadata.get("parent_trace"),
            string_value(metadata.get("source_filename"), path.stem),
        )
        start = np.asarray(metadata.get("window_start", [[0]])).reshape(-1)
        window_start[index] = int(start[0]) if len(start) else 0

        if (index + 1) % 1000 == 0 or index + 1 == count:
            print(f"converted {index + 1}/{count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=x,
        y=y,
        participant=participant,
        monitor=monitor,
        repetition=repetition,
        source=source,
        window_start=window_start,
        activity_names=np.asarray(list("ABCDEFGHIJKLMNOPQRST")),
        angle_names=ANGLE_NAMES,
        window_size=np.asarray(10, dtype=np.int16),
        stride=np.asarray(10, dtype=np.int16),
        representation=np.asarray("BFA recovered from complex V"),
    )
    total_vectors = count * 10 * 234
    print("Saved:", args.output)
    print("x:", x.shape, x.dtype, "y:", y.shape)
    print("mean unit-norm error before projection:", norm_error_sum / total_vectors)
    print("maximum unit-norm error before projection:", norm_error_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validation = subparsers.add_parser("validate")
    validation.add_argument("--vmatrix", type=Path, required=True)
    validation.add_argument("--bfa", type=Path, required=True)
    validation.add_argument("--max-frames", type=int)
    validation.set_defaults(func=validate)

    conversion = subparsers.add_parser("convert-mat")
    conversion.add_argument("input", type=Path)
    conversion.add_argument("output", type=Path)
    conversion.add_argument(
        "--metadata-dir",
        type=Path,
        help=(
            "paired real MAT root with matching relative filenames; use its "
            "parent_trace/source_filename/window_start when generated MAT files "
            "did not preserve those fields"
        ),
    )
    conversion.set_defaults(func=convert_mat)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
