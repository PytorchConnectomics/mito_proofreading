"""Index low-res slices for the configured neuron ids (per-slice bbox + tile occupancy).

Usage: python scripts/02_neuron_index.py configs/h01.yaml [job_id job_num]
Job ``job_id`` handles low-res z-tiles tz with tz % job_num == job_id and writes
{output_root}/index/part_{job_id}_{job_num}.npz (skipped if present).
"""
import os
import sys

import numpy as np

from mito_proofreading.chunks import part_path
from mito_proofreading.config import load_config
from mito_proofreading.neurons import index_slices, job_slices, load_neuron_ids


def main(argv):
    cfg = load_config(argv[1])
    job_id, job_num = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (0, 1)
    path = part_path(cfg, job_id, job_num)
    if os.path.exists(path):
        print(f"exists, skipping: {path}")
        return 0
    ids = load_neuron_ids(cfg)
    z_list = job_slices(cfg, job_id, job_num)
    print(f"job {job_id}/{job_num}: {len(ids)} neurons, {len(z_list)} slices", flush=True)
    slice_rows, tile_rows = index_slices(cfg, ids, z_list, log=lambda m: print(m, flush=True))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez(tmp, ids=ids, z_list=np.asarray(z_list, np.int64),
             slice_rows=slice_rows, tile_rows=tile_rows)
    os.replace(tmp, path)
    print(f"wrote {path}: {len(slice_rows)} slice rows, {len(tile_rows)} tile rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
