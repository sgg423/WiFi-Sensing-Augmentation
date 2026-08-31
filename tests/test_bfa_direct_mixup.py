import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/augment_bfa_circular_mixup.py'
spec = importlib.util.spec_from_file_location('mixup', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MixupTests(unittest.TestCase):
    def test_boundaries(self):
        a = np.broadcast_to([511, 0, 0, 127], (1, 10, 234, 4))
        b = np.broadcast_to([1, 510, 127, 0], a.shape)
        out = module.mix_angles(a, b, [0.25])
        self.assertTrue(np.all(out[..., 0] >= 511))
        self.assertTrue(np.all((out[..., 1] == 0) | (out[..., 1] == 511)))
        self.assertTrue(np.all(out[..., 2] == 32))
        self.assertTrue(np.all(out[..., 3] == 95))
        np.testing.assert_array_equal(module.mix_angles(a, b, [0]), a)
        # A constant input sequence stays constant: no independent frame noise.
        np.testing.assert_array_equal(out[:, 0], out[:, -1])

    def test_cli_train_only_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(1)
            x = rng.integers(0, 128, (80, 10, 234, 4), dtype=np.uint16)
            y = np.arange(80) % 4
            source = np.array([f'trace{i}.pcapng' for i in range(80)])
            np.savez(root / 'real.npz', x=x, y=y, source=source, window_start=np.arange(80)*10)
            command = [sys.executable, str(SCRIPT), str(root / 'real.npz')]
            for name in ('a.npz', 'b.npz'):
                subprocess.run(command + [str(root / name), '--train-split-seed', '111'], check=True, capture_output=True)
            with np.load(root / 'a.npz') as a, np.load(root / 'b.npz') as b:
                mask = np.random.default_rng(111).random(80) < .7
                np.testing.assert_array_equal(a['augmentation_eligible'], mask)
                np.testing.assert_array_equal(a['x'][~mask], x[~mask])
                np.testing.assert_array_equal(a['source'], source)
                np.testing.assert_array_equal(a['window_start'], np.arange(80)*10)
                np.testing.assert_array_equal(a['y'], y)
                self.assertTrue(mask[a['mixup_peer'][mask]].all())
                self.assertTrue((a['mixup_peer'][mask] != np.arange(80)[mask]).all())
                np.testing.assert_array_equal(y[a['mixup_peer']], y)
                np.testing.assert_array_equal(a['x'], b['x'])
            failed = subprocess.run(command + [str(root / 'a.npz'), '--train-split-seed', '111'], capture_output=True)
            self.assertNotEqual(failed.returncode, 0)


if __name__ == '__main__':
    unittest.main()
