"""Download the low-res segmentation crop as per-slice h5 + legacy bbox files.

Usage: python scripts/01_download_lowres_seg.py configs/h01.yaml [job_id job_num]
Job ``job_id`` handles download slabs i with i % job_num == job_id.
Complete slices (seg and bbox present) are skipped; nothing is overwritten.
"""
import sys

from mito_proofreading.config import load_config
from mito_proofreading.lowres import download_slab, slab_ranges
from mito_proofreading.volume import open_precomputed


def main(argv):
    cfg = load_config(argv[1])
    job_id, job_num = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (0, 1)
    store = open_precomputed(cfg["gs"]["seg"], cfg["lowres"]["resolution_xyz"])
    for i, (z0, z1) in enumerate(slab_ranges(cfg)):
        if i % job_num != job_id:
            continue
        written = download_slab(cfg, z0, z1, store=store)
        print(f"slab crop z [{z0}, {z1}): wrote {len(written)} slices", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
