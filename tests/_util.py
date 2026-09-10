"""Shared helpers for the synthetic unit tests."""
import copy
import os

import numpy as np
import yaml

from mito_proofreading.config import load_config, lowres_path
from mito_proofreading.h5io import write_h5_atomic
from mito_proofreading.lowres import slice_bbox

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H01_YAML = os.path.join(REPO, "configs", "h01.yaml")

# Low-res 32x32x40 nm, inference 8x8x20 nm -> ratio zyx [2, 4, 4]; mask 16x16x20 -> [1, 2, 2].
BASE = {
    "name": "synthetic",
    "data_root": None,
    "output_root": "out",
    "gs": {"image": "gs://unused/image/", "seg": "gs://unused/seg/"},
    "lowres": {"resolution_xyz": [32, 32, 40], "size_xyz": [120, 90, 6],
               "crop_offset_zyx": [1, 10, 20], "crop_size_zyx": [4, 64, 80],
               "seg_pattern": "seg/%04d.h5", "bbox_pattern": "bbox/%04d_bb.h5",
               "download_block_yx": [32, 32], "download_zslab": 2, "h5_chunks_yx": [16, 16]},
    "inference": {"resolution_xyz": [8, 8, 20], "size_xyz": [470, 360, 11]},
    "tile": {"size_lowres_zyx": [2, 16, 16], "origin_lowres_zyx": [1, 10, 20],
             "min_voxels_lowres": 1, "dilate": 0},
    "chunk": {"pad_zyx": [0, 0, 0]},
    "neurons": {"pair_files": [], "ids": [5]},
    "mask": {"source": "lowres", "resolution_xyz": [16, 16, 20], "min_overlap": 0.5,
             "pred_pattern": "{output_root}/pred/{z0:04d}/{y0}-{x0}.h5",
             "out_pattern": "{output_root}/mito/{neuron_id}/{z0:04d}/{y0}-{x0}.h5"},
}


def make_cfg(tmpdir, **sections):
    """Synthetic config rooted at ``tmpdir``; dict overrides merge one level deep."""
    raw = copy.deepcopy(BASE)
    raw["data_root"] = tmpdir
    for key, value in sections.items():
        if isinstance(value, dict):
            raw[key].update(value)
        else:
            raw[key] = value
    path = os.path.join(tmpdir, "cfg.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(raw, f)
    return load_config(path)


def write_lowres(cfg, vol):
    """Write a crop volume (z, y, x) as legacy per-slice seg + bbox files."""
    for z in range(vol.shape[0]):
        write_h5_atomic(lowres_path(cfg, "seg_pattern", z), {"main": vol[z]})
        write_h5_atomic(lowres_path(cfg, "bbox_pattern", z), {"main": slice_bbox(vol[z])})


BIG_ID = 2 ** 33 + 7


def synthetic_volume():
    """Crop volume (4, 64, 80) uint64 with two query neurons and one distractor."""
    vol = np.zeros((4, 64, 80), np.uint64)
    vol[0:3, 5:20, 30:50] = 5          # spans tile borders in z, y, and x
    vol[0, 50:52, 2:4] = 5             # far fragment: bbox covers tiles the neuron does not touch
    vol[1:4, 40:60, 60:75] = BIG_ID    # id above 2**32
    vol[:, 0:4, 0:4] = 9               # not queried
    return vol
