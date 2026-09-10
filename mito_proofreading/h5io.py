"""HDF5 helpers: atomic writes (temp file + rename) that also handle empty arrays."""
import os

import h5py
import numpy as np


def write_h5_atomic(path, datasets):
    """Write ``{name: array}`` to ``path`` through ``path + ".tmp"`` and a rename."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        for name, arr in datasets.items():
            arr = np.asarray(arr)
            options = {"compression": "gzip"} if arr.size else {}
            f.create_dataset(name, data=arr, **options)
    os.replace(tmp, path)


def read_h5_dataset(path, name="main"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        return f[name][()]
