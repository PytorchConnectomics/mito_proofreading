"""Box and tile coordinate math.

A box is a length-6 int64 array ``[z0, z1, y0, y1, x0, x1]``: ZYX, half-open,
global (uncropped) coordinates at a stated resolution.
"""
import numpy as np


def as_box(box):
    return np.asarray(box, dtype=np.int64).reshape(6).copy()


def box_shape(box):
    box = as_box(box)
    return box[1::2] - box[0::2]


def box_is_empty(box):
    return bool(np.any(box_shape(box) <= 0))


def scale_box(box, ratio_zyx):
    """Coarse box -> fine box for an integer per-axis ratio."""
    return as_box(box) * np.repeat(np.asarray(ratio_zyx, np.int64), 2)


def downscale_box(box, ratio_zyx):
    """Fine box -> smallest coarse box covering it (floor start, ceil stop)."""
    box = as_box(box)
    ratio = np.asarray(ratio_zyx, np.int64)
    out = box.copy()
    out[0::2] = box[0::2] // ratio
    out[1::2] = -(-box[1::2] // ratio)
    return out


def clip_box(box, size_zyx, offset_zyx=(0, 0, 0)):
    """Clip a box to the region ``[offset, offset + size)``."""
    box = as_box(box)
    lo = np.asarray(offset_zyx, np.int64)
    hi = lo + np.asarray(size_zyx, np.int64)
    box[0::2] = np.clip(box[0::2], lo, hi)
    box[1::2] = np.clip(box[1::2], lo, hi)
    return box


def pad_box(box, pad_zyx):
    box = as_box(box)
    pad = np.asarray(pad_zyx, np.int64)
    box[0::2] -= pad
    box[1::2] += pad
    return box


def legacy_rows_to_global(rows, z, crop_offset_zyx):
    """Legacy per-slice bbox rows -> global half-open rows.

    Input rows are ``[id, y0, y1, x0, x1, count]`` with inclusive max in crop
    coordinates of slice ``z`` (as written by ``em_util.io.compute_bbox_all``).
    Output rows are ``[id, z0, z1, y0, y1, x0, x1, count]`` in global low-res voxels.
    """
    rows = np.asarray(rows, np.int64).reshape(-1, 6)
    off = np.asarray(crop_offset_zyx, np.int64)
    out = np.empty((len(rows), 8), np.int64)
    out[:, 0] = rows[:, 0]
    out[:, 1] = z + off[0]
    out[:, 2] = z + off[0] + 1
    out[:, 3] = rows[:, 1] + off[1]
    out[:, 4] = rows[:, 2] + off[1] + 1
    out[:, 5] = rows[:, 3] + off[2]
    out[:, 6] = rows[:, 4] + off[2] + 1
    out[:, 7] = rows[:, 5]
    return out


def tile_index(coord, origin, size):
    """Tile index of global low-res coordinates along one or more axes."""
    return (np.asarray(coord, np.int64) - origin) // size


def tile_box_lowres(tile_zyx, cfg):
    """Unclipped global low-res box of tile ``(tz, ty, tx)``."""
    geom = cfg["geom"]
    start = geom["tile_origin_zyx"] + np.asarray(tile_zyx, np.int64) * geom["tile_size_zyx"]
    box = np.empty(6, np.int64)
    box[0::2] = start
    box[1::2] = start + geom["tile_size_zyx"]
    return box


def tile_to_inference_box(tile_zyx, cfg):
    """``(out_box, in_box)`` at inference resolution for a low-res tile.

    out_box: the tile clipped to the low-res crop (where the neuron index exists),
    scaled to inference resolution and clipped to the inference volume.
    in_box: out_box padded by ``chunk.pad_zyx`` and clipped to the inference volume.
    """
    geom = cfg["geom"]
    lowres = clip_box(tile_box_lowres(tile_zyx, cfg), geom["crop_size_zyx"], geom["crop_offset_zyx"])
    out_box = clip_box(scale_box(lowres, geom["ratio_zyx"]), geom["inference_size_zyx"])
    in_box = clip_box(pad_box(out_box, geom["pad_zyx"]), geom["inference_size_zyx"])
    return out_box, in_box


def chunk_key(box):
    """Legacy chunk name ``{z0:04d}/{y0}-{x0}`` of an inference-resolution box."""
    box = as_box(box)
    return f"{int(box[0]):04d}/{int(box[2])}-{int(box[4])}"
