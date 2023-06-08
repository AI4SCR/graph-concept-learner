#!/usr/bin/env python3

import athena as ath
import torch
from torch_geometric.utils.convert import from_networkx
import os.path as osp
import yaml
import sys
import pickle
import pandas as pd

(
    prog_name,
    spl,
    sample_ids_file,
    cfg_file,
    so_file,
    output_file,
) = sys.argv

# Get df with the labels and its sample_ids
prediction_labels = pd.read_csv(sample_ids_file, index_col=0).squeeze("columns")

# Load so obj
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Load config
with open(cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

# Generate graphs for config
concept_name = cfg["concept_name"]
builder_type = cfg["builder_type"]

# Make a graph for the given sample, and turn it into pyg and save to file
# Build graph
ath.graph.build_graph(
    so, spl, builder_type=builder_type, config=cfg, key_added=concept_name
)

# Remove edge weights
for (n1, n2, d) in so.G[spl][concept_name].edges(data=True):
    d.clear()

# From netx to pyg
g = from_networkx(
    G=so.G[spl][concept_name], group_node_attrs=all, group_edge_attrs=None
)

# Attach label
g.y = torch.tensor([prediction_labels[spl]])

# Name file
path_name = osp.join(output_file)

# Save to file
torch.save(g, path_name)
