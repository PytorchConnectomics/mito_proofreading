import tempfile
import unittest

import numpy as np

from _util import make_cfg, synthetic_volume, write_lowres
from mito_proofreading import mask as mask_module
from mito_proofreading.mask import check_pred, mask_mito_chunk, neuron_mask


def three_mito():
    seg = np.zeros((1, 10, 10), np.uint32)
    seg[0, 0, :] = 1   # 10 voxels, all inside
    seg[0, 2, :] = 2   # 10 voxels, 6 inside
    seg[0, 4, :] = 3   # 10 voxels, 4 inside
    mask = np.zeros(seg.shape, bool)
    mask[0, 0, :] = True
    mask[0, 2, :6] = True
    mask[0, 4, :4] = True
    return seg, mask


class TestMask(unittest.TestCase):
    def check(self, seg, mask, ids):
        masked, table = mask_mito_chunk(seg, mask, 0.5)
        self.assertEqual(masked.dtype, seg.dtype)
        np.testing.assert_array_equal(np.unique(masked), [0, ids[0], ids[1]])
        np.testing.assert_array_equal(masked[seg == ids[1]], ids[1])  # kept whole, not clipped
        np.testing.assert_array_equal(table, [[ids[0], 10, 10], [ids[1], 10, 6], [ids[2], 10, 4]])

    def test_overlap_threshold(self):
        seg, mask = three_mito()
        self.check(seg, mask, [1, 2, 3])
        self.check(seg.astype(np.uint64), mask, [1, 2, 3])

    def test_large_ids_use_unique(self):
        seg, mask = three_mito()
        big = seg.astype(np.uint64) + np.uint64(2 ** 40) * (seg > 0)
        ids = [2 ** 40 + 1, 2 ** 40 + 2, 2 ** 40 + 3]
        self.check(big, mask, ids)

    def test_slabs_match_single_pass(self):
        rng = np.random.default_rng(0)
        seg = rng.integers(0, 6, (7, 5, 4)).astype(np.uint32)
        mask = rng.random((7, 5, 4)) > 0.4
        expected = mask_mito_chunk(seg, mask, 0.5)
        saved = mask_module.SLAB_VOXELS
        mask_module.SLAB_VOXELS = 30  # one z-slice per pass
        try:
            for got_arr, exp_arr in zip(mask_mito_chunk(seg, mask, 0.5), expected):
                np.testing.assert_array_equal(got_arr, exp_arr)
            big = seg.astype(np.uint64) + np.uint64(2 ** 40) * (seg > 0)
            masked, table = mask_mito_chunk(big, mask, 0.5)
            np.testing.assert_array_equal(masked > 0, expected[0] > 0)
            np.testing.assert_array_equal(table[:, 1:], expected[1][:, 1:])
        finally:
            mask_module.SLAB_VOXELS = saved

    def test_empty_chunk(self):
        masked, table = mask_mito_chunk(np.zeros((2, 3, 4), np.uint16), np.ones((2, 3, 4), bool), 0.5)
        self.assertFalse(masked.any())
        self.assertEqual(table.shape, (0, 3))

    def test_check_pred(self):
        box = [0, 2, 0, 3, 0, 4]
        check_pred(np.zeros((2, 3, 4), np.uint32), box)
        with self.assertRaises(ValueError):
            check_pred(np.zeros((2, 3, 5), np.uint32), box)
        with self.assertRaises(ValueError):
            check_pred(np.zeros((2, 3, 4), np.float32), box)
        with self.assertRaises(ValueError):
            check_pred(-np.ones((2, 3, 4), np.int32), box)

    def test_lowres_neuron_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            vol = synthetic_volume()
            write_lowres(cfg, vol)
            geom = cfg["geom"]
            full = np.zeros(geom["lowres_size_zyx"], np.uint64)
            off, size = geom["crop_offset_zyx"], geom["crop_size_zyx"]
            full[off[0]:off[0] + size[0], off[1]:off[1] + size[1], off[2]:off[2] + size[2]] = vol
            box = np.array([1, 9, 41, 150, 83, 231])  # unaligned, starts below the crop in z
            zz, yy, xx = np.meshgrid(np.arange(box[0], box[1]), np.arange(box[2], box[3]),
                                     np.arange(box[4], box[5]), indexing="ij")
            ratio = geom["ratio_zyx"]
            expected = full[zz // ratio[0], yy // ratio[1], xx // ratio[2]] == 5
            got = neuron_mask(cfg, 5, box)
            np.testing.assert_array_equal(got, expected)
            self.assertTrue(got.any() and not got[0].any())


if __name__ == "__main__":
    unittest.main()
