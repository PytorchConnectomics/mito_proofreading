"""Tensorstore helpers for neuroglancer precomputed volumes (e.g. public H01 on GCS)."""
import numpy as np


def precomputed_spec(url, resolution_xyz):
    """Tensorstore spec selecting one scale of a precomputed volume."""
    return {
        "driver": "neuroglancer_precomputed",
        "kvstore": url,
        "scale_metadata": {"resolution": [float(r) for r in resolution_xyz]},
    }


def open_precomputed(url, resolution_xyz, context=None):
    """Open one scale read-only; public buckets need no credentials."""
    import tensorstore as ts

    return ts.open(precomputed_spec(url, resolution_xyz), read=True, context=context).result()


def read_box(store, box):
    """Read a ZYX half-open box ``[z0, z1, y0, y1, x0, x1]`` as a ZYX array."""
    z0, z1, y0, y1, x0, x1 = (int(v) for v in box)
    data = store[x0:x1, y0:y1, z0:z1, 0].read().result()
    return np.ascontiguousarray(np.asarray(data).transpose(2, 1, 0))
