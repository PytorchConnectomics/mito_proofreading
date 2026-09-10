"""Neuron-mask decoding: keep predicted mito instances that lie inside each chunk's neurons.

Usage: python scripts/04_mask_mito.py configs/h01.yaml [job_id job_num]
Job ``job_id`` handles rows i of chunks_all.csv with i % job_num == job_id. Each
prediction (mask.pred_pattern, see mito_proofreading/mask.py for the contract) is
read once and masked for every neuron of that chunk; outputs go to
mask.out_pattern (datasets main, table) and existing outputs are skipped.
"""
import os
import sys

from mito_proofreading.chunks import read_chunks_all
from mito_proofreading.config import load_config, mask_path
from mito_proofreading.h5io import read_h5_dataset, write_h5_atomic
from mito_proofreading.mask import check_pred, mask_mito_chunk, neuron_mask, open_mask_store


def main(argv):
    cfg = load_config(argv[1])
    job_id, job_num = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (0, 1)
    min_overlap = float(cfg["mask"]["min_overlap"])
    store = open_mask_store(cfg)
    written = missing = failed = 0
    for i, row in enumerate(read_chunks_all(cfg)):
        if i % job_num != job_id:
            continue
        box = row["out_box"]
        fields = {"z0": box[0], "y0": box[2], "x0": box[4]}
        outputs = {n: mask_path(cfg, "out_pattern", neuron_id=n, **fields) for n in row["neuron_ids"]}
        todo = [n for n, path in outputs.items() if not os.path.exists(path)]
        if not todo:
            continue
        pred_path = mask_path(cfg, "pred_pattern", **fields)
        if not os.path.exists(pred_path):
            missing += 1
            continue
        pred = read_h5_dataset(pred_path, "main")
        try:
            check_pred(pred, box)
        except ValueError as err:
            print(f"{row['chunk_key']}: invalid prediction {pred_path}: {err}", flush=True)
            failed += 1
            continue
        for n in todo:
            masked, table = mask_mito_chunk(pred, neuron_mask(cfg, n, box, store), min_overlap)
            write_h5_atomic(outputs[n], {"main": masked, "table": table})
            kept = int((table[:, 2] >= min_overlap * table[:, 1]).sum())
            print(f"{row['chunk_key']} neuron {n}: kept {kept}/{len(table)} mito", flush=True)
            written += 1
    print(f"wrote {written} outputs; {missing} chunks without prediction; {failed} invalid predictions")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
