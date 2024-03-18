from ai4bmr_core.log.log import logger
from pathlib import Path

import pandas as pd


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

    # align all indices
    idx = expr.index
    lab = lab.loc[idx]
    loc = loc.loc[idx]
    spat = spat.loc[idx]

    lab.drop(columns=["CellId"]).to_parquet(labels_path)
    expr.drop(columns=["CellId"]).to_parquet(expr_path)
    loc.to_parquet(loc_path)
    spat.drop(columns=["CellId"]).to_parquet(spat_path)

    if removed_masks:
        logger.info(f"Removing masks_ids from {mask.stem}: {removed_masks}")
        for mask_id in removed_masks:
            mask[mask == mask_id] = 0

        imsave(mask_path, mask, plugin="tifffile")
