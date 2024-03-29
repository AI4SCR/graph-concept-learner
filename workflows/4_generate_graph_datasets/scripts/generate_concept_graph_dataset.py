#!/usr/bin/env python3

import athena as ath
import os.path as osp
import os
import yaml
import sys
import pickle
import pandas as pd

(
    prog_name,
    sample_ids_file,
    cfg_file,
    so_file,
    output_dir,
) = sys.argv

# Load list of filtered samples and prediction labels
prediction_labels = pd.read_csv(sample_ids_file, index_col=0).squeeze("columns")
all_samples = prediction_labels.index.values

# Load so obj
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Load config
with open(cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

# Delete pre-loaded graphs
so.G.clear()

# Generate graphs for config
concept_name = cfg["concept_name"]
builder_type = cfg["builder_type"]

# Do not attribute
cfg["build_and_attribute"] = False

# Make directory where the graphs will be written
os.makedirs(output_dir, exist_ok=True)

# Make graph for every sample, turn it into pyg and save to file
for spl in all_samples:
    # Extract centroid
    ath.pp.extract_centroids(so, spl, mask_key="cellmasks")

    randomize = cfg.pop("randomize", {})
    randomize_cell_labels = randomize.pop("cell_labels", False)
    if randomize_cell_labels:
        randomize_seed = randomize.pop("seed", 42)
        col_name = cfg["concept_params"]["filter_col"]
        so.obs[spl][col_name] = so.obs[spl][col_name].sample(
            frac=1, replace=False, random_state=randomize_seed
        )

    # Build graph
    ath.graph.build_graph(so, spl, config=cfg, key_added=concept_name)

    # Remove edge weights
    for (n1, n2, d) in so.G[spl][concept_name].edges(data=True):
        d.clear()

    # Name file
    path_name = osp.join(output_dir, f"{spl}.pkl")

    # Write to output
    with open(path_name, "wb") as f:
        pickle.dump(so.G[spl][concept_name], f)
