import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "vmatrix_to_bfa.py"
SPEC = importlib.util.spec_from_file_location("vmatrix_to_bfa", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_quantization_centers_round_trip():
    rng = np.random.default_rng(42)
    bfa = np.column_stack(
        (
            rng.integers(0, 512, 10000),
            rng.integers(0, 512, 10000),
            rng.integers(0, 128, 10000),
            rng.integers(0, 128, 10000),
        )
    ).astype(np.uint16)
    reconstructed = MODULE.bfa_to_vmatrix(bfa)
    recovered = MODULE.vmatrix_to_bfa(reconstructed)
    np.testing.assert_array_equal(recovered, bfa)


def test_inverse_is_invariant_to_norm_and_global_phase():
    bfa = np.asarray([[400, 425, 37, 26]], dtype=np.uint16)
    reconstructed = MODULE.bfa_to_vmatrix(bfa)
    changed = reconstructed * 2.5 * np.exp(1j * 1.234)
    np.testing.assert_array_equal(MODULE.vmatrix_to_bfa(changed), bfa)
