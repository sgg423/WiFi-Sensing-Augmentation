"""Exercise the actual dry-run/data functions without requiring PyTorch/CUDA.

Only function definitions are loaded; torch.manual_seed is a no-op for dry-run.
No model construction, optimization, or GPU behavior is simulated/tested here.
"""

import ast
import contextlib
import hashlib
import io
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


SCRIPT = Path(__file__).parents[1] / "scripts/train_sensefi_har1.py"
tree = ast.parse(SCRIPT.read_text())
functions = ast.Module(
    body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[]
)
namespace = dict(
    np=np, h5py=h5py, Path=Path, hashlib=hashlib, json=json, random=random,
    argparse=__import__("argparse"), train_test_split=train_test_split,
    torch=types.SimpleNamespace(manual_seed=lambda seed: None), __file__=str(SCRIPT),
)
exec(compile(functions, str(SCRIPT), "exec"), namespace)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.real = self.root / "real.h5"
        self.synthetic = self.root / "synthetic.h5"
        self.order = np.random.default_rng(9).permutation(400)
        for path, order in ((self.real, np.arange(400)), (self.synthetic, self.order)):
            with h5py.File(path, "w") as handle:
                handle.create_dataset("x", shape=(400, 1, 250, 242), dtype="f2")
                handle["y"] = (order // 20).astype("i2")
                handle["participant"] = np.ones(400, dtype="i2")
                handle["window_start"] = order % 4 * 250
                handle.create_dataset("source", data=np.array(
                    [f"user{i // 4:04d}.mat" for i in order], dtype=object
                ), dtype=h5py.string_dtype("utf-8"))

    def run_selection(self, name, seed, augment=True, split_seed=111, augment_seed=111, ratio=1.0):
        output = self.root / name
        argv = [str(SCRIPT), "--data", str(self.real), "--output-dir", str(output),
                "--seed", str(seed), "--split-seed", str(split_seed),
                "--augment-seed", str(augment_seed), "--dry-run"]
        if augment:
            argv += ["--augment-data", str(self.synthetic), "--augment-ratio", str(ratio),
                     "--augment-match", "train-window"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            namespace["main"]()
        with np.load(output / "selection_indices.npz") as handle:
            indices = {key: handle[key].copy() for key in handle.files}
        return indices, json.loads((output / "protocol.json").read_text())

    def test_fixed_data_across_model_seeds_and_baseline(self):
        baseline, _ = self.run_selection("base", 42, augment=False)
        first, config = self.run_selection("aug42", 42)
        second, _ = self.run_selection("aug111", 111)
        for field in ("train", "validation", "test"):
            np.testing.assert_array_equal(baseline[field], first[field])
            np.testing.assert_array_equal(first[field], second[field])
        np.testing.assert_array_equal(first["synthetic"], second["synthetic"])
        np.testing.assert_array_equal(self.order[first["synthetic"]], first["train"])
        self.assertEqual((len(first["train"]), len(first["validation"]), len(first["test"])), (280, 60, 60))
        self.assertFalse(set(self.order[first["synthetic"]]) & set(first["test"]))
        self.assertEqual(config["split_seed"], 111)

    def test_augment_seed_changes_only_synthetic_subset(self):
        first, _ = self.run_selection("a", 42, ratio=0.5)
        second, _ = self.run_selection("b", 42, augment_seed=42, ratio=0.5)
        for field in ("train", "validation", "test"):
            np.testing.assert_array_equal(first[field], second[field])
        self.assertNotEqual(set(first["synthetic"]), set(second["synthetic"]))
        self.assertEqual(len(first["synthetic"]), 140)

    def test_split_seed_changes_split(self):
        first, _ = self.run_selection("a", 42, augment=False)
        second, _ = self.run_selection("b", 42, augment=False, split_seed=42)
        self.assertNotEqual(set(first["test"]), set(second["test"]))

    def test_missing_provenance_fails(self):
        with h5py.File(self.synthetic, "a") as handle:
            del handle["window_start"]
        with self.assertRaisesRegex(ValueError, "window_start"):
            self.run_selection("bad", 111)

    def test_missing_matches_fail(self):
        with h5py.File(self.synthetic, "a") as handle:
            handle["window_start"][:] += 9999
        with self.assertRaisesRegex(ValueError, "Missing .* generated train windows"):
            self.run_selection("bad", 111)

    def test_duplicate_keys_fail(self):
        with h5py.File(self.synthetic, "a") as handle:
            for field in ("source", "window_start", "y"):
                handle[field][1] = handle[field][0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_selection("bad", 111)


if __name__ == "__main__":
    unittest.main()
