"""Dataset configuration: every data path and dataset fact comes from one yaml.

Yaml resolutions and sizes are XYZ (neuroglancer order). Derived geometry in
``cfg["geom"]`` is ZYX, matching numpy volume axes.
"""
import os

import numpy as np
import yaml

REQUIRED = {
    None: ["name", "data_root", "output_root", "gs", "lowres", "inference",
           "tile", "chunk", "neurons", "mask"],
    "gs": ["image", "seg"],
    "lowres": ["resolution_xyz", "size_xyz", "crop_offset_zyx", "crop_size_zyx",
               "seg_pattern", "bbox_pattern", "download_block_yx", "download_zslab",
               "h5_chunks_yx"],
    "inference": ["resolution_xyz", "size_xyz"],
    "tile": ["size_lowres_zyx", "origin_lowres_zyx", "min_voxels_lowres", "dilate"],
    "chunk": ["pad_zyx"],
    "neurons": ["pair_files", "ids"],
    "mask": ["source", "resolution_xyz", "min_overlap", "pred_pattern", "out_pattern"],
}
MASK_SOURCES = ("gs", "lowres")


def integer_ratio(coarse_zyx, fine_zyx, name):
    """Per-axis ratio coarse/fine; raise unless it is an integer >= 1."""
    ratio = np.asarray(coarse_zyx, float) / np.asarray(fine_zyx, float)
    rounded = np.round(ratio)
    if np.any(np.abs(ratio - rounded) > 1e-6) or np.any(rounded < 1):
        raise ValueError(f"config: {name} resolution ratio {ratio.tolist()} is not an integer >= 1")
    return rounded.astype(np.int64)


def _check_keys(cfg):
    for section, keys in REQUIRED.items():
        node = cfg if section is None else cfg.get(section)
        if not isinstance(node, dict):
            raise ValueError(f"config: missing section '{section}'")
        missing = [k for k in keys if k not in node]
        if missing:
            raise ValueError(f"config: section '{section or 'top'}' is missing keys {missing}")


def _vec(values, name, length=3):
    arr = np.asarray(values, dtype=np.int64)
    if arr.shape != (length,):
        raise ValueError(f"config: {name} needs {length} integers, got {values}")
    return arr


def load_config(path):
    """Load and validate a dataset yaml; add derived ZYX geometry under ``geom``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config: {path} is not a mapping")
    _check_keys(cfg)
    lowres, inference, mask = cfg["lowres"], cfg["inference"], cfg["mask"]
    if mask["source"] not in MASK_SOURCES:
        raise ValueError(f"config: mask.source must be one of {MASK_SOURCES}")
    if not 0 < float(mask["min_overlap"]) <= 1:
        raise ValueError("config: mask.min_overlap must be in (0, 1]")

    lowres_res = np.asarray(lowres["resolution_xyz"], float)[::-1].copy()
    inference_res = np.asarray(inference["resolution_xyz"], float)[::-1].copy()
    mask_res = np.asarray(mask["resolution_xyz"], float)[::-1].copy()
    geom = {
        "lowres_resolution_zyx": lowres_res,
        "lowres_size_zyx": _vec(lowres["size_xyz"], "lowres.size_xyz")[::-1].copy(),
        "inference_resolution_zyx": inference_res,
        "inference_size_zyx": _vec(inference["size_xyz"], "inference.size_xyz")[::-1].copy(),
        "mask_resolution_zyx": mask_res,
        "ratio_zyx": integer_ratio(lowres_res, inference_res, "lowres/inference"),
        "mask_ratio_zyx": integer_ratio(mask_res, inference_res, "mask/inference"),
        "crop_offset_zyx": _vec(lowres["crop_offset_zyx"], "lowres.crop_offset_zyx"),
        "crop_size_zyx": _vec(lowres["crop_size_zyx"], "lowres.crop_size_zyx"),
        "tile_size_zyx": _vec(cfg["tile"]["size_lowres_zyx"], "tile.size_lowres_zyx"),
        "tile_origin_zyx": _vec(cfg["tile"]["origin_lowres_zyx"], "tile.origin_lowres_zyx"),
        "pad_zyx": _vec(cfg["chunk"]["pad_zyx"], "chunk.pad_zyx"),
    }
    crop_end = geom["crop_offset_zyx"] + geom["crop_size_zyx"]
    if np.any(geom["crop_offset_zyx"] < 0) or np.any(crop_end > geom["lowres_size_zyx"]):
        raise ValueError("config: low-res crop lies outside lowres.size_xyz")
    if np.any(geom["crop_size_zyx"] < 1) or np.any(geom["tile_size_zyx"] < 1):
        raise ValueError("config: crop and tile sizes must be positive")
    if np.any(geom["pad_zyx"] < 0) or int(cfg["tile"]["dilate"]) < 0:
        raise ValueError("config: chunk.pad_zyx and tile.dilate must be non-negative")
    block = _vec(lowres["download_block_yx"], "lowres.download_block_yx", 2)
    chunks = _vec(lowres["h5_chunks_yx"], "lowres.h5_chunks_yx", 2)
    if np.any(chunks < 1) or np.any(block % chunks):
        raise ValueError("config: lowres.h5_chunks_yx must divide lowres.download_block_yx")
    if int(lowres["download_zslab"]) < 1:
        raise ValueError("config: lowres.download_zslab must be positive")

    cfg["geom"] = geom
    cfg["config_path"] = os.path.abspath(path)
    cfg["output_dir"] = resolve(cfg, cfg["output_root"])
    return cfg


def resolve(cfg, path):
    """Absolute path; relative paths are taken relative to data_root."""
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(cfg["data_root"], path)


def lowres_path(cfg, key, z):
    """Per-slice low-res file from a printf pattern (z index in crop coordinates)."""
    return resolve(cfg, cfg["lowres"][key] % int(z))


def mask_path(cfg, key, **fields):
    """Path from a str.format pattern under ``mask``."""
    fields = {k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in fields.items()}
    return resolve(cfg, cfg["mask"][key].format(output_root=cfg["output_dir"], **fields))
