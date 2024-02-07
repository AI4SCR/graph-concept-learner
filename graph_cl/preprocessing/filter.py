import yaml
import pickle

import pandas as pd
from pathlib import Path


def filter_sample_has_target(valid_cores: set, spl: pd.DataFrame, target: str):
    spls_with_target = spl[spl[target].notna()].index
    valid_cores = valid_cores.intersection(spls_with_target)
    return valid_cores


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
    output: Path,
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

    so.spl.loc[list(valid_cores), [target, "cohort"]].rename(
        columns={target: "target"}
    ).to_csv(output)
