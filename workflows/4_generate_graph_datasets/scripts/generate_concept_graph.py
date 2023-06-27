#!/usr/bin/env python3

import athena as ath
import yaml
import sys
import pickle

(
    prog_name,
    spl,
    cfg_file,
    so_file,
    output_file,
) = sys.argv

# Load so obj
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Load config
with open(cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

# Generate graphs for config
concept_name = cfg["concept_name"]
builder_type = cfg["builder_type"]

# Do not attribute
cfg["build_and_attribute"] = False

# Make a graph for the given sample, and turn it into pyg and save to file
# Build graph
ath.graph.build_graph(
    so, spl, builder_type=builder_type, config=cfg, key_added=concept_name
)

# Remove edge weights
for (n1, n2, d) in so.G[spl][concept_name].edges(data=True):
    d.clear()

# Write to output
with open(output_file, "wb") as f:
    pickle.dump(so.G[spl][concept_name], f)
