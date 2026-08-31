import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1]/'scripts/train_timevae_bfa.py'
spec = importlib.util.spec_from_file_location('timevae_bfa', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class AdapterTests(unittest.TestCase):
    def test_quantized_roundtrip(self):
        x = np.random.default_rng(7).integers(0,128,(6,10,234,4),dtype=np.uint16)
        x[..., :2] *= 4
        x[0,0,0] = [511,0,127,0]
        encoded = m.encode(x)
        self.assertEqual(encoded.shape, (6,10,1404))
        np.testing.assert_array_equal(m.decode(encoded), x)

    def test_bounds_and_invalid(self):
        x = np.ones((2,10,1404), dtype=np.float32)*5
        out = m.decode(x)
        self.assertEqual(out.dtype, np.uint16)
        self.assertTrue((out[..., :2] <= 511).all())
        self.assertTrue((out[..., 2:] == 127).all())
        x[0,0,0] = np.nan
        with self.assertRaises(ValueError):
            m.decode(x)

    def test_split_matches_beamsense(self):
        mask = np.random.default_rng(111).random(40675) < .7
        np.testing.assert_array_equal(m.train_indices(40675,111), np.flatnonzero(mask))
        self.assertEqual(mask.sum(),28529)


if __name__ == '__main__':
    unittest.main()
