import logging

import yaml
import pickle

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def filter_sample_has_target(cores: set, spl: pd.DataFrame, target: str):
    spls_with_target = spl[spl[target].notna()].index
    cores = cores.intersection(spls_with_target)
    return cores


def filter_sample_has_min_cells(
    valid_cores: set, obs: pd.DataFrame, filter_col, labels, min_cells_per_graph: int
):
    spls_with_min_cells = set()
    for spl in valid_cores:
        if obs[spl][filter_col].isin(labels).sum() > min_cells_per_graph:
            spls_with_min_cells.add(spl)
    return valid_cores.intersection(spls_with_min_cells)


def filter_samples(
    so_path: Path,
    concepts_dir: Path,
    target: str,
    min_cells_per_graph: int,
    output_path: Path,
):
    # Load all configs (one for each concept)
    cfgs = []
    for p in concepts_dir.glob("*.yaml"):
        with open(p) as f:
            cfg = yaml.safe_load(f)
        cfgs.append(cfg)

    with open(so_path, "rb") as f:
        so = pickle.load(f)

    valid_cores = set(so.spl.index)

    for cfg in cfgs:
        # Unpack relevant config params
        if cfg["build_concept_graph"] is False:
            continue

        labels = cfg["concept_params"]["include_labels"]
        filter_col = cfg["concept_params"]["filter_col"]

        valid_cores = filter_sample_has_target(valid_cores, so.spl, target)
        valid_cores = filter_sample_has_min_cells(
            valid_cores, so.obs, filter_col, labels, min_cells_per_graph
        )

    meta = so.spl.loc[list(valid_cores), [target, "cohort"]]
    encoder = LabelEncoder()
    meta = meta.assign(target=encoder.fit_transform(meta[target]))
    meta.to_csv(output_path)


import numpy as np
import pandas as pd


def filter_mask_objects(mask: np.ndarray, labels: pd.Series, include_labels: list):
    m = labels.isin(include_labels)
    obj_ids = set(labels[m].index.get_level_values("cell_id"))
    mask_objs = set(mask.flatten())
    mask_objs.remove(0)

    for obj in mask_objs - obj_ids:
        mask[mask == obj] = 0

    return mask


def harmonize_index(
    mask_path: Path, expr_path: Path, labels_path: Path, loc_path: Path, spat_path: Path
):
    from skimage.io import imread, imsave

    mask = imread(mask_path, plugin="tifffile")
    lab = pd.read_parquet(labels_path)
    loc = pd.read_parquet(loc_path)
    spat = pd.read_parquet(spat_path)
    expr = pd.read_parquet(expr_path)

    assert len(lab) == len(expr)
    assert len(lab) == len(loc)
    assert len(lab) == len(spat)

    uniq_masks = set(mask.flatten())
    uniq_masks.remove(0)  # remove background

    uniq_loc = set(loc.ObjectNumber)

    removed_masks = uniq_masks - uniq_loc
    assert uniq_loc - uniq_masks == set()

    mapping = dict(zip(loc.ObjectNumber_renamed, loc.ObjectNumber))
    lab = lab.assign(cell_id=lab.CellId.map(mapping).astype(int)).set_index(
        ["core", "cell_id"], verify_integrity=True
    )
    assert lab.index.get_level_values("cell_id").isna().any() == False
    expr = expr.assign(cell_id=expr.CellId.map(mapping).astype(int)).set_index(
        ["core", "cell_id"], verify_integrity=True
    )
    assert expr.index.get_level_values("cell_id").isna().any() == False
    spat = spat.assign(cell_id=spat.CellId.map(mapping).astype(int)).set_index(
        ["core", "cell_id"], verify_integrity=True
    )
    assert spat.index.get_level_values("cell_id").isna().any() == False
    loc = loc.assign(cell_id=loc.ObjectNumber.astype(int)).set_index(
        ["core", "cell_id"], verify_integrity=True
    )
    assert loc.index.get_level_values("cell_id").isna().any() == False

    assert lab.isna().any().any() == False
    assert loc.isna().any().any() == False
    assert expr.isna().any().any() == False
    assert spat.isna().any().any() == False

    lab.drop(columns=["CellId"]).to_parquet(labels_path)
    expr.drop(columns=["CellId"]).to_parquet(expr_path)
    loc.to_parquet(loc_path)
    spat.drop(columns=["CellId"]).to_parquet(spat_path)

    if removed_masks:
        logging.info(f"Removing masks_ids from {mask.stem}: {removed_masks}")
        for mask_id in removed_masks:
            mask[mask == mask_id] = 0

        imsave(mask_path, mask, plugin="tifffile")
