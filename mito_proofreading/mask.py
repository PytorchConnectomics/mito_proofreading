"""Neuron masks for chunks and neuron-masked mito instance segmentation.

Prediction-file contract (DL package -> 04_mask_mito): one h5 per chunk at
``mask.pred_pattern`` with dataset ``main``: ZYX, non-negative integer instance
ids (0 = background), shape exactly the chunk out box at inference resolution,
even when the DL package read a padded input box.
"""
import numpy as np

from .geometry import as_box, box_is_empty, box_shape, clip_box, downscale_box
from .lowres import open_slice
from .volume import open_precomputed, read_box

BINCOUNT_MAX_ID = 50_000_000  # above this, count instances with np.unique instead
SLAB_VOXELS = 1 << 24  # voxels per pass in mask_mito_chunk (~128 MB of int64 temporaries)


def check_pred(pred, out_box):
    """Enforce the prediction-file contract; raise ValueError otherwise."""
    shape = tuple(int(s) for s in box_shape(out_box))
    if pred.ndim != 3 or tuple(pred.shape) != shape:
        raise ValueError(f"prediction shape {pred.shape} != out box shape {shape}")
    if not np.issubdtype(pred.dtype, np.integer):
        raise ValueError(f"prediction dtype {pred.dtype} is not an integer type")
    if pred.size and pred.min() < 0:
        raise ValueError("prediction contains negative instance ids")


def open_mask_store(cfg):
    """Tensorstore for ``mask.source == "gs"``; None for the local low-res source."""
    if cfg["mask"]["source"] != "gs":
        return None
    return open_precomputed(cfg["gs"]["seg"], cfg["mask"]["resolution_xyz"])


def _read_padded(store, box):
    """Read a ZYX box, zero-filling the part outside the store's domain."""
    size_zyx = np.asarray(store.domain.shape[:3], np.int64)[::-1]
    inside = clip_box(box, size_zyx)
    out = np.zeros(box_shape(box), store.dtype.numpy_dtype)
    if not box_is_empty(inside):
        start = inside[0::2] - box[0::2]
        stop = start + box_shape(inside)
        out[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = read_box(store, inside)
    return out


def _lowres_mask(cfg, neuron_id, coarse):
    """Neuron mask over a global low-res box from the local slice files (zero outside the crop)."""
    geom = cfg["geom"]
    offset = geom["crop_offset_zyx"]
    mask = np.zeros(box_shape(coarse), bool)
    crop = clip_box(coarse, geom["crop_size_zyx"], offset)
    if box_is_empty(crop):
        return mask
    for z in range(crop[0], crop[1]):
        with open_slice(cfg, z - offset[0]) as f:
            seg = f["main"][crop[2] - offset[1]:crop[3] - offset[1],
                            crop[4] - offset[2]:crop[5] - offset[2]]
        mask[z - coarse[0], crop[2] - coarse[2]:crop[3] - coarse[2],
             crop[4] - coarse[4]:crop[5] - coarse[4]] = seg == np.uint64(neuron_id)
    return mask


def _upsample_crop(coarse, ratio, start, shape):
    """Nearest-neighbor upsample by an integer ratio, then crop ``shape`` at ``start``."""
    for axis, r in enumerate(ratio):
        if r > 1:
            coarse = np.repeat(coarse, int(r), axis=axis)
    return coarse[tuple(slice(int(s), int(s) + int(n)) for s, n in zip(start, shape))]


def neuron_mask(cfg, neuron_id, box, store=None):
    """Boolean ZYX mask of ``neuron_id`` over an inference-resolution box."""
    box = as_box(box)
    geom = cfg["geom"]
    if cfg["mask"]["source"] == "gs":
        ratio = geom["mask_ratio_zyx"]
        coarse = downscale_box(box, ratio)
        if store is None:
            store = open_mask_store(cfg)
        mask = _read_padded(store, coarse) == np.uint64(neuron_id)
    else:
        ratio = geom["ratio_zyx"]
        coarse = downscale_box(box, ratio)
        mask = _lowres_mask(cfg, neuron_id, coarse)
    return _upsample_crop(mask, ratio, box[0::2] - coarse[0::2] * ratio, box_shape(box))


def _z_slabs(shape):
    step = max(1, SLAB_VOXELS // max(1, int(np.prod(shape[1:]))))
    return [slice(z, z + step) for z in range(0, shape[0], step)]


def mask_mito_chunk(mito_seg, neuron_mask, min_overlap):
    """Keep mito instances with at least ``min_overlap`` of their voxels inside the neuron.

    Kept instances are copied whole (not clipped to the neuron); others are zeroed.
    Returns ``(masked_seg, table)``; table rows are ``[mito_id, voxels, voxels_in_neuron]``.
    Works in z-slabs so temporaries stay small next to a full chunk.
    """
    if mito_seg.shape != neuron_mask.shape:
        raise ValueError(f"mito seg {mito_seg.shape} and neuron mask {neuron_mask.shape} differ")
    slabs = _z_slabs(mito_seg.shape)
    max_id = int(mito_seg.max()) if mito_seg.size else 0
    if max_id <= BINCOUNT_MAX_ID:
        labels = np.arange(max_id + 1)

        def index(seg):
            return seg.astype(np.int64)
    else:
        labels = np.unique(np.concatenate([np.unique(mito_seg[s]) for s in slabs]))

        def index(seg):
            return np.searchsorted(labels, seg)
    total = np.zeros(len(labels), np.int64)
    in_neuron = np.zeros(len(labels), np.int64)
    for s in slabs:
        idx = index(mito_seg[s]).ravel()
        total += np.bincount(idx, minlength=len(labels))
        in_neuron += np.bincount(idx[neuron_mask[s].ravel().astype(bool)], minlength=len(labels))
    present = (total > 0) & (labels > 0)
    keep = present & (in_neuron >= min_overlap * total)
    masked = np.zeros_like(mito_seg)
    for s in slabs:
        seg = mito_seg[s]
        selected = keep[index(seg)]
        masked[s][selected] = seg[selected]
    table = np.column_stack([labels[present].astype(np.int64), total[present], in_neuron[present]])
    return masked, table.reshape(-1, 3)
