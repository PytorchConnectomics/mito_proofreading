"""Neuron ids and the one-pass slice index (bbox rows + low-res tile occupancy)."""
import numpy as np

from .config import resolve
from .geometry import legacy_rows_to_global
from .lowres import open_slice, read_slice_bbox

MAX_EXACT_ID = 2 ** 53  # ids stay exact through any float64 promotion


def load_neuron_ids(cfg):
    """Sorted unique int64 ids from ``neurons.ids`` and ``neurons.pair_files``."""
    ids = [int(v) for v in (cfg["neurons"]["ids"] or [])]
    for rel in cfg["neurons"]["pair_files"] or []:
        with open(resolve(cfg, rel)) as f:
            for line in f:
                ids.extend(int(v) for v in line.split())
    if not ids:
        raise ValueError("no neuron ids: set neurons.ids or neurons.pair_files")
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if ids[0] <= 0 or ids[-1] >= MAX_EXACT_ID:
        raise ValueError(f"neuron ids must be in (0, 2**53), got [{ids[0]}, {ids[-1]}]")
    return ids


def job_slices(cfg, job_id, job_num):
    """Crop z indices of the low-res z-tiles ``tz`` with ``tz % job_num == job_id``."""
    geom = cfg["geom"]
    offset, depth = int(geom["crop_offset_zyx"][0]), int(geom["crop_size_zyx"][0])
    origin, size = int(geom["tile_origin_zyx"][0]), int(geom["tile_size_zyx"][0])
    return [z for z in range(depth) if ((z + offset - origin) // size) % job_num == job_id]


def index_slices(cfg, ids, z_list, log=None):
    """Index crop slices ``z_list`` for neuron ``ids``.

    Returns ``(slice_rows, tile_rows)``:
    slice_rows ``[id, z0, z1, y0, y1, x0, x1, count]``: global half-open low-res box per slice;
    tile_rows ``[id, tz, ty, tx, count]``: low-res voxels per occupied tile and slice.
    Each bbox file is read once; the union region of matched neurons is read once,
    in tile-row strips. Missing slice files raise.
    """
    geom = cfg["geom"]
    offset = geom["crop_offset_zyx"]
    origin, size = geom["tile_origin_zyx"], geom["tile_size_zyx"]
    ids = np.asarray(ids, np.int64)
    slice_rows, tile_rows = [], []
    for z in z_list:
        bbox = read_slice_bbox(cfg, z)
        rows = bbox[np.isin(bbox[:, 0], ids.astype(bbox.dtype))]
        if len(rows) == 0:
            continue
        rows = legacy_rows_to_global(rows, z, offset)
        slice_rows.append(rows)
        matched = np.unique(rows[:, 0])
        tz = (z + offset[0] - origin[0]) // size[0]
        y_lo, y_hi = rows[:, 3].min(), rows[:, 4].max()
        x_lo, x_hi = rows[:, 5].min(), rows[:, 6].max()
        tx_lo = (x_lo - origin[2]) // size[2]
        n_tx = (x_hi - 1 - origin[2]) // size[2] - tx_lo + 1
        ty_range = range((y_lo - origin[1]) // size[1], (y_hi - 1 - origin[1]) // size[1] + 1)
        with open_slice(cfg, z) as f:
            seg = f["main"]
            for ty in ty_range:
                s0 = max(origin[1] + ty * size[1], y_lo)
                s1 = min(origin[1] + (ty + 1) * size[1], y_hi)
                strip = seg[int(s0 - offset[1]):int(s1 - offset[1]),
                            int(x_lo - offset[2]):int(x_hi - offset[2])]
                hit = np.isin(strip, matched.astype(strip.dtype))
                if not hit.any():
                    continue
                cols = np.nonzero(hit)[1]
                id_index = np.searchsorted(matched, strip[hit].astype(np.int64))
                tx = (cols + x_lo - origin[2]) // size[2] - tx_lo
                keys, counts = np.unique(id_index * n_tx + tx, return_counts=True)
                tile_rows.append(np.column_stack([
                    matched[keys // n_tx], np.full(len(keys), tz), np.full(len(keys), ty),
                    keys % n_tx + tx_lo, counts]).astype(np.int64))
        if log:
            log(f"slice {z}: {len(matched)} neurons")
    slice_rows = np.concatenate(slice_rows) if slice_rows else np.zeros((0, 8), np.int64)
    tile_rows = np.concatenate(tile_rows) if tile_rows else np.zeros((0, 5), np.int64)
    return slice_rows, tile_rows
