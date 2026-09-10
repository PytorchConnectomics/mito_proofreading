import unittest

import numpy as np

from _util import H01_YAML
from mito_proofreading.config import load_config
from mito_proofreading.geometry import (chunk_key, clip_box, downscale_box, legacy_rows_to_global,
                                        pad_box, scale_box, tile_to_inference_box)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(H01_YAML)

    def test_legacy_rows(self):
        rows = legacy_rows_to_global([[7, 1, 3, 4, 9, 11]], 5, [0, 2560, 3520])
        np.testing.assert_array_equal(rows, [[7, 5, 6, 2561, 2564, 3524, 3530, 11]])

    def test_scale_downscale(self):
        np.testing.assert_array_equal(scale_box([1, 2, 3, 4, 5, 6], [4, 16, 16]),
                                      [4, 8, 48, 64, 80, 96])
        np.testing.assert_array_equal(downscale_box([5, 9, 17, 33, 0, 16], [4, 16, 16]),
                                      [1, 3, 1, 3, 0, 1])

    def test_pad_clip(self):
        box = pad_box([0, 10, 5, 20, 100, 200], [2, 10, 10])
        np.testing.assert_array_equal(box, [-2, 12, -5, 30, 90, 210])
        np.testing.assert_array_equal(clip_box(box, [11, 25, 205]), [0, 11, 0, 25, 90, 205])

    def test_tile_box_legacy_grid(self):
        out_box, in_box = tile_to_inference_box((24, 0, 0), self.cfg)
        np.testing.assert_array_equal(out_box, [2400, 2500, 40960, 43008, 56320, 58368])
        np.testing.assert_array_equal(in_box, out_box)
        self.assertEqual(chunk_key(out_box), "2400/40960-56320")

    def test_tile_clips(self):
        out_box, _ = tile_to_inference_box((52, 0, 0), self.cfg)
        np.testing.assert_array_equal(out_box[:2], [5200, 5293])  # low-res 1300..1324 -> 5296 -> 5293
        out_box, _ = tile_to_inference_box((0, 0, 211), self.cfg)
        np.testing.assert_array_equal(out_box[4:], [488448, 489472])  # crop x end 30592 * 16

    def test_padded_in_box(self):
        self.cfg["geom"]["pad_zyx"] = np.array([4, 64, 64])
        out_box, in_box = tile_to_inference_box((0, 0, 0), self.cfg)
        np.testing.assert_array_equal(in_box, [0, 104, 40896, 43072, 56256, 58432])
        self.assertEqual(out_box[0], 0)


if __name__ == "__main__":
    unittest.main()
