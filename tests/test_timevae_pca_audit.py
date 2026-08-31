import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_timevae_pca_bfa import restore_pca, choose_baseline
from train_timevae_bfa import encode


class AuditTests(unittest.TestCase):
    def test_saved_transform_is_not_refit(self):
        rng = np.random.default_rng(7)
        x = rng.integers(0,128,(8,10,234,4),dtype=np.uint16)
        pca = PCA(n_components=4,random_state=7).fit(encode(x).reshape(-1,1404))
        before = pca.components_.copy()
        scale = np.ones(4,dtype=np.float32)*2
        restored = restore_pca(x,dict(pca=pca,scale=scale))
        self.assertEqual(restored.shape,x.shape)
        self.assertEqual(restored.dtype,np.uint16)
        np.testing.assert_array_equal(before,pca.components_)
        self.assertTrue((restored[...,2:]<=127).all())
        with self.assertRaises(ValueError):
            restore_pca(x,dict(pca=pca,scale=np.zeros(4)))

    def test_baseline_selection_rejects_augmented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            real=root/'real.npz'
            for name,augmentation in [('base',None),('aug','generated.npz')]:
                folder=root/name
                folder.mkdir()
                (folder/'random_window_best.keras').touch()
                (folder/'random_window_metrics.json').write_text(json.dumps(dict(
                    real_data=str(real),augmentation_data=augmentation,split='random-window',
                    seed=111,split_seed=111)))
            self.assertEqual(choose_baseline(root,real,111,111),root/'base')
            with self.assertRaises(ValueError):
                choose_baseline(root,real,42,111)


if __name__ == '__main__':
    unittest.main()
