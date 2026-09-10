"""Check a dataset yaml against the live precomputed volumes and the local low-res crop.

Usage: python scripts/00_check_config.py configs/h01.yaml
"""
import sys

import numpy as np

from mito_proofreading.config import load_config
from mito_proofreading.lowres import read_slice_crop
from mito_proofreading.volume import open_precomputed, read_box

SPOT = 128  # half-width of the crop-offset spot check, low-res voxels


def main(argv):
    cfg = load_config(argv[1])
    geom = cfg["geom"]
    for key, value in geom.items():
        print(f"{key}: {np.asarray(value).tolist()}")
    errors = []
    checks = [
        ("lowres seg", cfg["gs"]["seg"], cfg["lowres"]["resolution_xyz"], cfg["lowres"]["size_xyz"]),
        ("inference seg", cfg["gs"]["seg"], cfg["inference"]["resolution_xyz"], cfg["inference"]["size_xyz"]),
        ("inference image", cfg["gs"]["image"], cfg["inference"]["resolution_xyz"], cfg["inference"]["size_xyz"]),
    ]
    stores = {}
    for name, url, resolution, size in checks:
        stores[name] = open_precomputed(url, resolution)
        shape = [int(s) for s in stores[name].domain.shape[:3]]
        ok = shape == [int(s) for s in size]
        print(f"{name} {resolution}: gs size_xyz {shape}, yaml {list(size)} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            errors.append(name)
    mask_store = open_precomputed(cfg["gs"]["seg"], cfg["mask"]["resolution_xyz"])
    print(f"mask seg {cfg['mask']['resolution_xyz']}: gs size_xyz {list(mask_store.domain.shape[:3])}")

    z = int(geom["crop_size_zyx"][0]) // 2
    cy, cx = (int(v) // 2 for v in geom["crop_size_zyx"][1:])
    y0, y1 = max(cy - SPOT, 0), min(cy + SPOT, int(geom["crop_size_zyx"][1]))
    x0, x1 = max(cx - SPOT, 0), min(cx + SPOT, int(geom["crop_size_zyx"][2]))
    local = read_slice_crop(cfg, z, y0, y1, x0, x1)
    off = geom["crop_offset_zyx"]
    remote = read_box(stores["lowres seg"], [z + off[0], z + off[0] + 1, y0 + off[1], y1 + off[1],
                                             x0 + off[2], x1 + off[2]])[0]
    match = float((local == remote).mean())
    print(f"crop offset spot check z={z} y[{y0},{y1}) x[{x0},{x1}): match {match:.4f}, "
          f"nonzero {float((local > 0).mean()):.3f}")
    if match < 1:
        errors.append("crop offset")
    if not local.any():
        print("warning: spot check region is empty; the match is uninformative")
    if errors:
        print("FAILED:", ", ".join(errors))
        return 1
    print("config OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
