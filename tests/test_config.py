import os
import tempfile
import unittest

import numpy as np

from _util import H01_YAML, make_cfg
from mito_proofreading.config import load_config, mask_path
from mito_proofreading.lowres import slab_ranges
from mito_proofreading.neurons import load_neuron_ids


class TestConfig(unittest.TestCase):
    def test_h01_geometry(self):
        cfg = load_config(H01_YAML)
        geom = cfg["geom"]
        np.testing.assert_array_equal(geom["ratio_zyx"], [4, 16, 16])
        np.testing.assert_array_equal(geom["mask_ratio_zyx"], [1, 4, 4])
        np.testing.assert_array_equal(geom["lowres_size_zyx"], [1324, 22275, 32244])
        np.testing.assert_array_equal(geom["inference_size_zyx"], [5293, 356400, 515892])
        self.assertEqual(cfg["output_dir"], os.path.join(cfg["data_root"], "mito_pf"))

    def test_rejects_bad_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                make_cfg(tmp, lowres={"h5_chunks_yx": [24, 24]})
            with self.assertRaises(ValueError):
                make_cfg(tmp, inference={"resolution_xyz": [7, 8, 20]})
            with self.assertRaises(ValueError):
                make_cfg(tmp, mask={"source": "local"})
            with self.assertRaises(ValueError):
                make_cfg(tmp, lowres={"crop_size_zyx": [6, 64, 80]})

    def test_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            path = mask_path(cfg, "out_pattern", neuron_id=5, z0=np.int64(100), y0=8, x0=np.int64(16))
            self.assertEqual(path, os.path.join(tmp, "out", "mito", "5", "0100", "8-16.h5"))

    def test_slab_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)  # crop z [1, 5) global, slab 2
            self.assertEqual(slab_ranges(cfg), [(0, 1), (1, 3), (3, 4)])

    def test_neuron_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pairs.txt"), "w") as f:
                f.write("590612150\t36750893213\n36750893213\t7\n")
            cfg = make_cfg(tmp, neurons={"pair_files": ["pairs.txt"], "ids": [7, 11]})
            np.testing.assert_array_equal(load_neuron_ids(cfg), [7, 11, 590612150, 36750893213])
            cfg = make_cfg(tmp, neurons={"pair_files": [], "ids": [2 ** 53]})
            with self.assertRaises(ValueError):
                load_neuron_ids(cfg)


if __name__ == "__main__":
    unittest.main()
