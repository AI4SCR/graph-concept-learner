#!/usr/bin/env python3
from athena.attributer.node_features import add_node_features
import pandas as pd
import os
import yaml
import sys
import pickle
import torch
from torch_geometric.utils.convert import from_networkx

# Debug input
# so_path = "/cluster/scratch/scastro/jackson/prediction_tasks/ERStatus/normalized_with_min_max/meta_data/normalized_data/fold_0.pkl"
# concept_path = "/cluster/scratch/scastro/jackson/prediction_tasks/ERStatus/normalized_with_min_max/processed_data/unattributed/all_cells_radius"
# config_path = "/cluster/scratch/scastro/jackson/prediction_tasks/ERStatus/normalized_with_min_max/configs/attribute_configs/ER.yaml"
# labels_path = "/cluster/scratch/scastro/jackson/prediction_tasks/ERStatus/normalized_with_min_max/meta_data/filtered_sample_ids_and_labels.csv"
# output_dir = "/cluster/scratch/scastro/jackson/prediction_tasks/ERStatus/normalized_with_min_max/processed_data/attributed/all_cells_radius_ER/fold_0/"

(
    prog_name,
    so_path,
    concept_path,
    config_path,
    labels_path,
    output_dir,
) = sys.argv

# Load config
with open(config_path) as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

# Load fold specific so object
with open(so_path, "rb") as f:
    so = pickle.load(f)

# Get df with the labels and its sample_ids
prediction_labels = pd.read_csv(labels_path, index_col=0).squeeze("columns")

# Load graphs and put them is so with the corresponding cocnept name and spl id
spls = []
concept_name = os.path.basename(concept_path)
for f in os.listdir(concept_path):
    if os.path.splitext(f)[1] == ".pkl":
        # Save name to the list of samples
        spl = os.path.splitext(f)[0]
        spls.append(spl)
        # Load graph
        graph_path = os.path.join(concept_path, f)
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        # Save in so object
        so.G[spl] = {}
        so.G[spl][concept_name] = graph

# Make directory where the graphs will be written
os.makedirs(output_dir, exist_ok=True)

# For every spl id, attribute the graph using attribute functionality form athena
# and save such graph to file usign torch geometric
for spl in spls:
    add_node_features(
        so=so,
        spl=spl,
        graph_key=concept_name,
        features_type=cfg["attrs_type"],
        config=cfg,
    )
    # From netx to pyg
    g = from_networkx(G=so.G[spl][concept_name], group_node_attrs=all)
    # Attach label
    g.y = torch.tensor([prediction_labels[spl]])
    # Name file
    attributed_graph_path = os.path.join(output_dir, f"{spl}.pt")
    # Save to file
    torch.save(g, attributed_graph_path)
