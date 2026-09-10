"""Merge index parts into per-neuron bboxes, chunk lists, and the global chunk manifest.

Usage: python scripts/03_neuron_chunks.py configs/h01.yaml job_num [partial]
Requires all parts part_{i}_{job_num}.npz from 02 unless 'partial' is given, in
which case present parts are merged and a PARTIAL marker is written.
"""
import os
import sys

import numpy as np

from mito_proofreading.chunks import merge_parts, part_path, write_manifests
from mito_proofreading.config import load_config
from mito_proofreading.neurons import load_neuron_ids


def main(argv):
    cfg = load_config(argv[1])
    job_num = int(argv[2])
    allow_partial = len(argv) > 3 and argv[3] == "partial"
    paths = [part_path(cfg, i, job_num) for i in range(job_num)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing and not allow_partial:
        print(f"missing {len(missing)}/{job_num} index parts, e.g. {missing[0]}; "
              "run 02 for them or pass 'partial'")
        return 1
    parts = [dict(np.load(p)) for p in paths if os.path.exists(p)]
    if not parts:
        print("no index parts found")
        return 1
    ids, slice_rows, tile_rows = merge_parts(parts)
    if not np.array_equal(ids, load_neuron_ids(cfg)):
        print(f"index parts were built for different neuron ids than the config; "
              f"delete {os.path.dirname(paths[0])} and rerun 02")
        return 1
    summary, chunks_all = write_manifests(cfg, ids, slice_rows, tile_rows, partial=bool(missing))
    empty = [row[0] for row in summary if row[1] == 0]
    print(f"{len(ids)} neurons, {len(chunks_all)} unique chunks, "
          f"{sum(row[-2] for row in summary)} neuron-chunk pairs"
          + (f", PARTIAL ({len(parts)}/{job_num} parts)" if missing else ""))
    if empty:
        print(f"neurons with no voxels in the indexed slices: {empty}")
    print(f"wrote {cfg['output_dir']}/chunks_all.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
