"""Merge index parts into per-neuron bboxes, chunk lists, and manifests."""
import csv
import os

import numpy as np

from .geometry import box_is_empty, chunk_key, clip_box, scale_box, tile_to_inference_box

BOX_COLUMNS = ["z0", "z1", "y0", "y1", "x0", "x1"]
IN_COLUMNS = ["in_" + c for c in BOX_COLUMNS]


def part_path(cfg, job_id, job_num):
    return os.path.join(cfg["output_dir"], "index", f"part_{job_id}_{job_num}.npz")


def merge_parts(parts):
    """Merge index parts (dicts with ``ids``, ``slice_rows``, ``tile_rows``).

    Parts must share the same query ids. Tile counts are summed over slices.
    """
    ids = np.asarray(parts[0]["ids"], np.int64)
    for part in parts[1:]:
        if not np.array_equal(np.asarray(part["ids"], np.int64), ids):
            raise ValueError("index parts were built for different neuron ids; delete stale parts")
    slice_rows = np.concatenate([np.asarray(p["slice_rows"], np.int64).reshape(-1, 8) for p in parts])
    tiles = np.concatenate([np.asarray(p["tile_rows"], np.int64).reshape(-1, 5) for p in parts])
    if len(tiles):
        keys, inverse = np.unique(tiles[:, :4], axis=0, return_inverse=True)
        counts = np.bincount(inverse.ravel(), weights=tiles[:, 4], minlength=len(keys))
        tiles = np.column_stack([keys, np.round(counts).astype(np.int64)])
    return ids, slice_rows, tiles


def neuron_bbox(slice_rows, neuron_id):
    """``(box, voxels, n_slices)`` of one neuron; box is global low-res half-open or None."""
    rows = slice_rows[slice_rows[:, 0] == neuron_id]
    if len(rows) == 0:
        return None, 0, 0
    box = np.array([rows[:, 1].min(), rows[:, 2].max(), rows[:, 3].min(),
                    rows[:, 4].max(), rows[:, 5].min(), rows[:, 6].max()], np.int64)
    return box, int(rows[:, 7].sum()), len(np.unique(rows[:, 1]))


def neuron_tiles(tile_rows, neuron_id, cfg):
    """``{(tz, ty, tx): voxels}`` after the min-voxel filter; dilated tiles get 0 voxels."""
    keep = (tile_rows[:, 0] == neuron_id) & (tile_rows[:, 4] >= int(cfg["tile"]["min_voxels_lowres"]))
    counts = {tuple(int(v) for v in row[1:4]): int(row[4]) for row in tile_rows[keep]}
    dilate = int(cfg["tile"]["dilate"])
    if dilate:
        steps = range(-dilate, dilate + 1)
        for tz, ty, tx in list(counts):
            for dz in steps:
                for dy in steps:
                    for dx in steps:
                        counts.setdefault((tz + dz, ty + dy, tx + dx), 0)
    return dict(sorted(counts.items()))


def neuron_chunks(tile_counts, cfg):
    """``[(chunk_key, out_box, in_box, voxels)]`` at inference resolution; empty boxes dropped."""
    chunks = []
    for tile, voxels in tile_counts.items():
        out_box, in_box = tile_to_inference_box(tile, cfg)
        if not box_is_empty(out_box):
            chunks.append((chunk_key(out_box), out_box, in_box, voxels))
    return chunks


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(path + ".tmp", path)


def write_manifests(cfg, ids, slice_rows, tile_rows, partial=False):
    """Write per-neuron bbox/chunks, ``neurons.csv``, ``chunks_all.csv``, and the PARTIAL marker.

    Returns ``(summary_rows, chunks_all)`` with ``chunks_all[key] = [out_box, in_box, neuron_ids]``.
    """
    out_dir = cfg["output_dir"]
    geom = cfg["geom"]
    summary, chunks_all = [], {}
    for nid in (int(v) for v in ids):
        box, voxels, n_slices = neuron_bbox(slice_rows, nid)
        chunks = neuron_chunks(neuron_tiles(tile_rows, nid, cfg), cfg)
        lines = ["# global half-open zyx boxes: z0 z1 y0 y1 x0 x1",
                 f"voxels_lowres {voxels}", f"slices_lowres {n_slices}"]
        box_fields = [""] * 12
        if box is not None:
            inference_box = clip_box(scale_box(box, geom["ratio_zyx"]), geom["inference_size_zyx"])
            lines += ["lowres " + " ".join(str(int(v)) for v in box),
                      "inference " + " ".join(str(int(v)) for v in inference_box)]
            box_fields = [int(v) for v in box] + [int(v) for v in inference_box]
        neuron_dir = os.path.join(out_dir, "neuron", str(nid))
        _write_text(os.path.join(neuron_dir, "bbox.txt"), "\n".join(lines) + "\n")
        _write_csv(os.path.join(neuron_dir, "chunks.csv"),
                   ["chunk_key"] + BOX_COLUMNS + IN_COLUMNS + ["voxels_lowres"],
                   [[key, *map(int, out_box), *map(int, in_box), voxels_tile]
                    for key, out_box, in_box, voxels_tile in chunks])
        for key, out_box, in_box, _ in chunks:
            chunks_all.setdefault(key, [out_box, in_box, set()])[2].add(nid)
        summary.append([nid, voxels, n_slices, *box_fields, len(chunks), int(partial)])

    _write_csv(os.path.join(out_dir, "neurons.csv"),
               ["neuron_id", "voxels_lowres", "slices_lowres"]
               + ["lowres_" + c for c in BOX_COLUMNS] + ["inference_" + c for c in BOX_COLUMNS]
               + ["n_chunks", "partial"], summary)
    _write_csv(os.path.join(out_dir, "chunks_all.csv"),
               ["chunk_key"] + BOX_COLUMNS + IN_COLUMNS + ["neuron_ids"],
               [[key, *map(int, out_box), *map(int, in_box), ";".join(str(n) for n in sorted(nids))]
                for key, (out_box, in_box, nids) in sorted(chunks_all.items())])
    marker = os.path.join(out_dir, "PARTIAL")
    if partial:
        _write_text(marker, "index merged from a subset of parts; rerun 02/03 for full results\n")
    elif os.path.exists(marker):
        os.remove(marker)
    return summary, chunks_all


def read_chunks_all(cfg):
    """Rows of ``chunks_all.csv`` as dicts with ``chunk_key``, ``out_box``, ``in_box``, ``neuron_ids``."""
    rows = []
    with open(os.path.join(cfg["output_dir"], "chunks_all.csv"), newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "chunk_key": row["chunk_key"],
                "out_box": np.array([int(row[c]) for c in BOX_COLUMNS], np.int64),
                "in_box": np.array([int(row[c]) for c in IN_COLUMNS], np.int64),
                "neuron_ids": [int(v) for v in row["neuron_ids"].split(";") if v],
            })
    return rows
