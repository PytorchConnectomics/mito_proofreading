"""Low-res segmentation slices on disk: download from GCS and per-slice IO.

Slice files keep the legacy layout: one h5 per crop z index with dataset
``main`` of shape ``crop_size_yx``, plus a bbox file with rows
``[id, y0, y1, x0, x1, count]`` (inclusive max, crop coordinates).
"""
import os

import h5py
import numpy as np

from .config import lowres_path
from .h5io import write_h5_atomic
from .volume import open_precomputed, read_box

BBOX_COLUMNS = 6


def slab_ranges(cfg):
    """Crop z ranges ``[z0, z1)`` of download slabs aligned to ``download_zslab`` in global z."""
    geom = cfg["geom"]
    offset, depth = int(geom["crop_offset_zyx"][0]), int(geom["crop_size_zyx"][0])
    slab = int(cfg["lowres"]["download_zslab"])
    ranges = []
    for start in range((offset // slab) * slab, offset + depth, slab):
        z0, z1 = max(start, offset), min(start + slab, offset + depth)
        if z0 < z1:
            ranges.append((z0 - offset, z1 - offset))
    return ranges


def slice_done(cfg, z):
    """A slice is complete only when both its seg and bbox files exist."""
    return (os.path.exists(lowres_path(cfg, "seg_pattern", z))
            and os.path.exists(lowres_path(cfg, "bbox_pattern", z)))


def slice_bbox(seg):
    """Legacy bbox rows of one slice via em_util (empty array for an empty slice)."""
    from em_util.io import compute_bbox_all

    rows = compute_bbox_all(seg, True)
    if rows is None:
        return np.zeros((0, BBOX_COLUMNS), np.int64)
    return np.asarray(rows, np.int64)


def download_slab(cfg, z0, z1, store=None, log=print):
    """Download incomplete crop slices in ``[z0, z1)``; return the z indices written.

    Reads c3 in ``download_block_yx`` blocks spanning all needed slices, writes each
    block into per-slice temp h5 files (h5py with ``h5_chunks_yx``, since
    ``em_util.io.write_h5`` cannot write partial blocks), then per slice writes the
    bbox file and renames the seg file last, so a present seg implies a present bbox.
    """
    todo = [z for z in range(z0, z1) if not slice_done(cfg, z)]
    if not todo:
        return []
    geom = cfg["geom"]
    offset, size = geom["crop_offset_zyx"], geom["crop_size_zyx"]
    height, width = int(size[1]), int(size[2])
    block_y, block_x = (int(v) for v in cfg["lowres"]["download_block_yx"])
    chunks = tuple(int(min(c, s)) for c, s in zip(cfg["lowres"]["h5_chunks_yx"], (height, width)))
    if store is None:
        store = open_precomputed(cfg["gs"]["seg"], cfg["lowres"]["resolution_xyz"])
    za, zb = todo[0], todo[-1] + 1
    log(f"download: reading crop z [{za}, {zb}) = {zb - za} slices, writing {len(todo)}")

    tmp = {z: lowres_path(cfg, "seg_pattern", z) + ".tmp" for z in todo}
    files = {}
    try:
        for z in todo:
            os.makedirs(os.path.dirname(tmp[z]), exist_ok=True)
            files[z] = h5py.File(tmp[z], "w")
            files[z].create_dataset("main", shape=(height, width), dtype=store.dtype.numpy_dtype,
                                    chunks=chunks, compression="gzip")
        for y in range(0, height, block_y):
            y1 = min(y + block_y, height)
            for x in range(0, width, block_x):
                x1 = min(x + block_x, width)
                box = [offset[0] + za, offset[0] + zb, offset[1] + y, offset[1] + y1,
                       offset[2] + x, offset[2] + x1]
                data = read_box(store, box)
                for z in todo:
                    files[z]["main"][y:y1, x:x1] = data[z - za]
    finally:
        for f in files.values():
            f.close()

    for z in todo:
        with h5py.File(tmp[z], "r") as f:
            seg = f["main"][()]
        write_h5_atomic(lowres_path(cfg, "bbox_pattern", z), {"main": slice_bbox(seg)})
        os.replace(tmp[z], lowres_path(cfg, "seg_pattern", z))
    return todo


def open_slice(cfg, z):
    """Open the seg h5 of crop slice ``z`` (caller closes); missing files raise."""
    path = lowres_path(cfg, "seg_pattern", z)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing low-res seg slice {path}")
    return h5py.File(path, "r")


def read_slice_crop(cfg, z, y0, y1, x0, x1):
    """Crop ``[y0, y1) x [x0, x1)`` (crop coordinates) of slice ``z``."""
    with open_slice(cfg, z) as f:
        return f["main"][int(y0):int(y1), int(x0):int(x1)]


def read_slice_bbox(cfg, z):
    """Legacy bbox rows of slice ``z``; missing files raise, empty files give 0 rows."""
    path = lowres_path(cfg, "bbox_pattern", z)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing low-res bbox file {path}")
    with h5py.File(path, "r") as f:
        names = list(f.keys())
        rows = f["main" if "main" in f else names[0]][()] if names else np.zeros(0)
    rows = np.asarray(rows)
    if rows.size == 0:
        return np.zeros((0, BBOX_COLUMNS), np.int64)
    if rows.ndim != 2 or rows.shape[1] != BBOX_COLUMNS:
        raise ValueError(f"{path}: expected rows of {BBOX_COLUMNS} columns, got {rows.shape}")
    return rows
