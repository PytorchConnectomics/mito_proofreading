import csv
import os
import tempfile
import unittest

import numpy as np

from _util import BIG_ID, make_cfg, synthetic_volume, write_lowres
from mito_proofreading.chunks import merge_parts, neuron_bbox, read_chunks_all, write_manifests
from mito_proofreading.config import lowres_path
from mito_proofreading.neurons import index_slices, job_slices


def brute_force(cfg, vol, ids):
    geom = cfg["geom"]
    off, origin, size = geom["crop_offset_zyx"], geom["tile_origin_zyx"], geom["tile_size_zyx"]
    boxes, tiles = {}, {}
    for nid in ids:
        zz, yy, xx = np.nonzero(vol == np.uint64(nid))
        coords = np.stack([zz, yy, xx], 1) + off
        boxes[nid] = (np.array([coords[:, 0].min(), coords[:, 0].max() + 1, coords[:, 1].min(),
                                coords[:, 1].max() + 1, coords[:, 2].min(), coords[:, 2].max() + 1]),
                      len(zz))
        keys, counts = np.unique((coords - origin) // size, axis=0, return_counts=True)
        tiles[nid] = {tuple(int(v) for v in k): int(c) for k, c in zip(keys, counts)}
    return boxes, tiles


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name, neurons={"ids": [5, BIG_ID]})
        self.vol = synthetic_volume()
        write_lowres(self.cfg, self.vol)
        self.ids = np.array([5, BIG_ID], np.int64)

    def tearDown(self):
        self.tmp.cleanup()

    def index(self, job_num):
        parts = []
        for job_id in range(job_num):
            slice_rows, tile_rows = index_slices(self.cfg, self.ids, job_slices(self.cfg, job_id, job_num))
            parts.append({"ids": self.ids, "slice_rows": slice_rows, "tile_rows": tile_rows})
        return merge_parts(parts)

    def test_matches_brute_force(self):
        boxes, tiles = brute_force(self.cfg, self.vol, self.ids)
        for job_num in (1, 2):
            _, slice_rows, tile_rows = self.index(job_num)
            for nid in self.ids:
                box, voxels, _ = neuron_bbox(slice_rows, nid)
                np.testing.assert_array_equal(box, boxes[nid][0])
                self.assertEqual(voxels, boxes[nid][1])
                got = {tuple(int(v) for v in r[1:4]): int(r[4]) for r in tile_rows if r[0] == nid}
                self.assertEqual(got, tiles[nid])

    def test_job_slices_partition(self):
        slices = sorted(z for j in range(3) for z in job_slices(self.cfg, j, 3))
        self.assertEqual(slices, list(range(4)))
        self.assertEqual(job_slices(self.cfg, 1, 2), [2, 3])

    def test_missing_slice_raises(self):
        os.remove(lowres_path(self.cfg, "bbox_pattern", 2))
        with self.assertRaises(FileNotFoundError):
            index_slices(self.cfg, self.ids, [0, 1, 2, 3])

    def test_mismatched_parts_raise(self):
        rows = np.zeros((0, 8)), np.zeros((0, 5))
        with self.assertRaises(ValueError):
            merge_parts([{"ids": [5], "slice_rows": rows[0], "tile_rows": rows[1]},
                         {"ids": [6], "slice_rows": rows[0], "tile_rows": rows[1]}])

    def test_manifests(self):
        ids, slice_rows, tile_rows = self.index(1)
        write_manifests(self.cfg, np.append(ids, 123), slice_rows, tile_rows, partial=True)
        out = self.cfg["output_dir"]
        self.assertTrue(os.path.exists(os.path.join(out, "PARTIAL")))
        rows = read_chunks_all(self.cfg)
        _, tiles = brute_force(self.cfg, self.vol, self.ids)
        self.assertEqual(len(rows), len(set(tiles[5]) | set(tiles[BIG_ID])))
        first = rows[0]
        np.testing.assert_array_equal(first["out_box"], first["in_box"])
        # tile (0, 0, 1): low-res z [1, 3), y [10, 26), x [36, 52) -> inference x ratio [2, 4, 4]
        by_key = {r["chunk_key"]: r for r in rows}
        np.testing.assert_array_equal(by_key["0002/40-144"]["out_box"], [2, 6, 40, 104, 144, 208])
        self.assertEqual(by_key["0002/40-144"]["neuron_ids"], [5])
        with open(os.path.join(out, "neurons.csv")) as f:
            summary = {int(r["neuron_id"]): r for r in csv.DictReader(f)}
        self.assertEqual(summary[123]["voxels_lowres"], "0")
        self.assertEqual(summary[5]["partial"], "1")
        write_manifests(self.cfg, ids, slice_rows, tile_rows, partial=False)
        self.assertFalse(os.path.exists(os.path.join(out, "PARTIAL")))


if __name__ == "__main__":
    unittest.main()
