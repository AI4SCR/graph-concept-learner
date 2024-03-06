"""
Makes graphs using athena and saves them to as pytorch geometric graphs.
"""
import athena as ath
import numpy as np
from pandas.api.types import is_numeric_dtype
import torch
from torch_geometric.utils.convert import from_networkx
import os.path as osp
import os
import argparse
import json

# # This is just ofr debugging
# args = {'configs_dir':  '/Users/ast/Documents/GitHub/datasets/clinical_type/configs',
#         'datasets_dir': '/Users/ast/Documents/GitHub/datasets/clinical_type/',
#         'preiction_target': 'clinical_type'
# }

# Parse command line arguments to pass to the script
parser = argparse.ArgumentParser(
    description="Graph dataset generation script. Makes graphs using athena and saves them to as pytorch geometric graphs to `datasets_dir`",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("configs_dir", help="Path to dir with config files.")
parser.add_argument("datasets_dir", help="Path to dir where to write the datasets.")
parser.add_argument(
    "preiction_target",
    help="Column of `so.spl` with the labels on which to train the GNNs.",
)
args = parser.parse_args()
args = vars(args)

# %% Import config files
configs = []
for file in os.listdir(args["configs_dir"]):

    if file.endswith(".txt"):
        # Constructing full path
        file_path = osp.join(args["configs_dir"], file)

        # reading the data from the file
        with open(file_path) as f:
            data = f.read()

        # reconstructing the data as a dictionary
        js = json.loads(data)

        # append to list
        configs.append(js)

# Load data and clean it
## Import data from athena
so = ath.dataset.imc()

## List all the samples
all_samples = so.spl.index.values

## Define prediction target
preiction_target = args["preiction_target"]

# Print sample size before filtering
print(f"Filtering samples. Total sample size before filter: {all_samples.size}")

for config in configs:
    # Unpack relevant config params
    labels = config["concept_params"]["labels"]
    filter_col = config["concept_params"]["filter_col"]

    # Remove incompleate or not well defined samples
    for spl in all_samples:
        # Remove sample if type is not in the colum
        if not np.all(np.isin(labels, so.obs[spl][filter_col].values)):
            all_samples = np.delete(all_samples, np.where(all_samples == spl))

        # Remove sample if prediction label is nan
        if so.spl.loc[spl][preiction_target] is np.nan:
            all_samples = np.delete(all_samples, np.where(all_samples == spl))

print(f"Sample size after: {all_samples.size}")

# Make map from strings to numbers if the prediction target is a column of stings
if not is_numeric_dtype(so.spl[preiction_target]):
    # Get array of stings with unique lables
    keys = so.spl[preiction_target][all_samples].unique()
    numeric_labels = list(range(0, len(keys)))
    map_to_numeric = dict(zip(keys, numeric_labels))

    # Make new column from so.spl with the prediction label as a numeric value
    prediction_labels = so.spl[preiction_target][all_samples].map(map_to_numeric)
else:
    prediction_labels = so.spl[preiction_target][all_samples]

# Extract location for every remaining sample
for spl in all_samples:
    ath.pp.extract_centroids(so, spl, mask_key="cellmasks")

## Delete pre-loaded graphs
so.G.clear()

# Generate graphs for every config
for config in configs:
    concept_name = config["concept_name"]
    builder_type = config["builder_type"]

    # Generate graphs for each config and for every sample
    for spl in all_samples:
        # Build graph
        ath.graph.build_graph(
            so, spl, builder_type=builder_type, config=config, key_added=concept_name
        )

        # Remove edge weights
        for (n1, n2, d) in so.G[spl][concept_name].edges(data=True):
            d.clear()

# Save graphs as pytorch files
for config in configs:
    # Make directory where the graphs will be written
    concept_name = config["concept_name"]
    concept_dir = osp.join(args["datasets_dir"], concept_name, "data")
    os.makedirs(concept_dir, exist_ok=True)

    for spl in all_samples:
        file_name = spl + ".pt"
        path_name = osp.join(concept_dir, file_name)
        G = from_networkx(
            G=so.G[spl][concept_name], group_node_attrs=all, group_edge_attrs=None
        )
        G.y = torch.tensor([prediction_labels[spl]])
        torch.save(G, path_name)

print("Done!")
