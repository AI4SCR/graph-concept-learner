#!/usr/bin/env python3

import athena as ath
import torch
from torch_geometric.utils.convert import from_networkx
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

# sample_ids_file = snakemake.input[0]
# cfg_file = snakemake.input[1]
# so_file = snakemake.input[2]
# preiction_target = snakemake.config["preiction_target"]
# output_dir = snakemake.output[0]

# sample_ids_file = "/Users/ast/Documents/GitHub/datasets/ER/meta_data/filtered_sample_ids_and_labels.csv"
# preiction_target = "ER"
# cfg_file = "/Users/ast/Documents/GitHub/datasets/ER/configs/dataset_configs/immune_tumor_radius.yaml"
# so_file = "/Users/ast/Documents/GitHub/datasets/ER/int_data/full_so.pkl"
# output_dir = "/Users/ast/Documents/GitHub/datasets/ER/prd_data/immune_tumor_radius"

# Load list of filtered samples and prediction labels
prediction_labels = pd.read_csv(sample_ids_file, index_col=0).squeeze("columns")
all_samples = prediction_labels.index.values

# Load so obj
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Load config
with open(cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

## Delete pre-loaded graphs
so.G.clear()

# Generate graphs for config
concept_name = cfg["concept_name"]
builder_type = cfg["builder_type"]

# Make directory where the graphs will be written
os.makedirs(output_dir, exist_ok=True)

# %% Make graph for evry sample, turn it into pyg and save to file
for spl in all_samples:
    # Extract centroid
    ath.pp.extract_centroids(so, spl, mask_key="cellmasks")

    # Do not attribute
    cfg["build_and_attribute"] = False

    # Build graph
    ath.graph.build_graph(
        so, spl, builder_type=builder_type, config=cfg, key_added=concept_name
    )

    # Remove edge weights
    for (n1, n2, d) in so.G[spl][concept_name].edges(data=True):
        d.clear()

    # Name file
    path_name = osp.join(output_dir, f"{spl}.pt")

    # Write to output
    with open(path_name, "wb") as f:
        pickle.dump(so.G[spl][concept_name], f)
