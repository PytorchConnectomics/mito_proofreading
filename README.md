# mito_proofreading

Data preparation for large-scale mitochondria prediction and proofreading. First
target: all mito in many neuron pairs of the public H01 dataset.

Code and data are separate: every data path and dataset fact lives in one yaml
(`configs/h01.yaml`); the package and scripts contain no paths.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 0 | `scripts/00_check_config.py <yaml>` | yaml vs live GCS sizes; local crop offset spot check |
| 1 | `scripts/01_download_lowres_seg.py <yaml> [job_id job_num]` | `lowres.seg_pattern` / `lowres.bbox_pattern` per z slice (skips complete slices) |
| 2 | `scripts/02_neuron_index.py <yaml> [job_id job_num]` | `{output_root}/index/part_{job_id}_{job_num}.npz` |
| 3 | `scripts/03_neuron_chunks.py <yaml> job_num [partial]` | `neuron/<id>/bbox.txt`, `neuron/<id>/chunks.csv`, `neurons.csv`, `chunks_all.csv` |
| DL | external package | one mito instance seg per chunk at `mask.pred_pattern` |
| 4 | `scripts/04_mask_mito.py <yaml> [job_id job_num]` | `mask.out_pattern`: neuron-masked mito seg + instance table |

Step 2 reads each slice's bbox file once and the union region of the queried
neurons once, recording per-slice boxes and occupied low-res tiles. Step 3 turns
occupied tiles into chunk boxes and de-duplicates chunks shared by several
neurons, so the DL package runs once per unique chunk.

## Environment

```bash
cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/donglai/lib/mito_proofreading
PY=/n/home04/donglai/miniconda3/envs/tensorstore/bin/python   # tensorstore, h5py, numpy, pyyaml
export PYTHONPATH=$PWD:/n/home04/donglai/lib/seg/em_util
# or install: $PY -m pip install -e . && $PY -m pip install -e /n/home04/donglai/lib/seg/em_util
$PY -m unittest discover -s tests -v
```

`em_util` (bbox computation) is a prerequisite, not a declared dependency.
Public H01 buckets are read anonymously; tensorstore's credential warnings are harmless.

## Conventions

* Yaml `*_xyz` values use neuroglancer order; arrays and `*_zyx` values use numpy ZYX order.
* Boxes written by the pipeline are global (uncropped), half-open `[z0, z1, y0, y1, x0, x1]`
  at the resolution named in the file or column (`lowres` 128x128x132 nm, `inference` 8x8x33 nm for H01).
* Legacy per-slice bbox files keep their format: rows `[id, y0, y1, x0, x1, count]`,
  inclusive max, crop coordinates.
* Chunk key `{z0:04d}/{y0}-{x0}` (inference voxels) matches the legacy tile names, and the
  default tile grid (`tile.origin_lowres_zyx` = crop offset) matches legacy tiles.
* `chunks_all.csv` has the out box (`z0..x1`), the padded input box (`in_z0..in_x1`,
  `chunk.pad_zyx`, clipped to the volume), and the chunk's `neuron_ids` joined by `;`.
* `lowres.*_pattern` use printf with the crop z index; `mask.*_pattern` use `str.format`
  fields `{output_root}`, `{neuron_id}`, `{z0}`, `{y0}`, `{x0}`.

## Prediction contract (DL package -> step 4)

For each row of `chunks_all.csv`, read image from `gs.image` at `inference.resolution_xyz`
over the input box (e.g. `mito_proofreading.volume.open_precomputed` + `read_box`), then write
one h5 at `mask.pred_pattern` with dataset `main`: ZYX, non-negative integer instance ids,
shape exactly the out box `(z1-z0, y1-y0, x1-x0)`. Step 4 rejects anything else.
Alternatively, call `mito_proofreading.mask.neuron_mask` and `mask_mito_chunk` during
decoding to write only the masked output.

Step 4 keeps a mito instance for a neuron if at least `mask.min_overlap` of its voxels lie in
the neuron mask; kept instances are copied whole. Output datasets: `main` (masked seg) and
`table` (rows `[mito_id, voxels, voxels_in_neuron]`). The neuron mask comes from c3 at
`mask.resolution_xyz` (`source: gs`, default 32x32x33 nm) or from the local low-res slices
(`source: lowres`).

## Running on the cluster

All steps are CPU jobs. Example SLURM array for step 2 (53 low-res z-tiles in H01):

```bash
sbatch --array=0-52 -p shared -c 1 --mem=8G -t 0-04:00 --wrap \
  "$PY scripts/02_neuron_index.py configs/h01.yaml \$SLURM_ARRAY_TASK_ID 53"
$PY scripts/03_neuron_chunks.py configs/h01.yaml 53
```

Sizing: an H01 bbox file holds ~17M rows (~0.8 GB in memory); step 1 holds a
`download_zslab x download_block_yx` uint64 block (~2 GB) plus one full slice (~3.4 GB)
for the bbox. The existing H01 low-res slices under `data_root` are complete, so step 1
only matters for regeneration or a new dataset yaml. Step 2 on one z-tile (25 slices, one
neuron) took ~2 min and 1.9 GB. Step 4 peaks at ~4 GB for a uint32 100x2048x2048 chunk
(prediction + masked copy + neuron mask); login nodes cap a user at 8 GB total, so run
several step-4 jobs through SLURM rather than side by side on a login node.

## Open defaults

* `chunk.pad_zyx: [0, 0, 0]` matches the legacy runs; chunk borders split mito instances and
  cross-chunk stitching is not handled here.
* `tile.dilate: 0`: 128 nm occupancy can miss thin neurites; raise it to add neighbor tiles.
* Neuron ids must be c3 agglomeration ids; `03` lists neurons with no voxels.

## Legacy map (`eng/T_eva.py`)

| T_eva option | Now |
|---|---|
| `0` per-slice bbox | step 1 (bbox computed at download) |
| `0.12`, `0.121` z-range index | not needed: step 2 filters bbox files by the query ids |
| `0.2` id -> bbox and tiles | steps 2 + 3 (fixes the far-edge tile clip) |
| `1.1` tensorstore pkl | `gs.image` + `inference.resolution_xyz`, `volume.precomputed_spec` |
| `1.2` pytc SLURM jobs | out of scope (DL package) |
| `2` per-neuron folders | `{output_root}/neuron/<id>/`, `{output_root}/mito/<id>/` |
