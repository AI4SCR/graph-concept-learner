#!/usr/bin/env python3

import os
import os.path as osp
import yaml
import pickle
import numpy as np
import math
import sys
from pandas.api.types import is_numeric_dtype

# Read inputs
(
    prog_name,
    cfgs_dir,
    so_file,
    prediction_target,
    min_num_cells_per_graph,
    filtered_sample_ids_and_labels,
    label_dict,
) = sys.argv

# Cast to int
min_num_cells_per_graph = int(min_num_cells_per_graph)

# Load all configs (one for each concept)
cfgs = []
for file in os.listdir(cfgs_dir):

    if file.endswith(".yaml"):
        # Constructing full path
        file_path = osp.join(cfgs_dir, file)

        # reading the data from the file
        with open(file_path) as f:
            cfg = yaml.safe_load(file)

        # append to list
        cfgs.append(cfg)

# %% Load so object
with open(so_file, "rb") as f:
    so = pickle.load(f)

# List all the samples
all_samples = so.spl.index.values

# Print log file sample size before filtering
print(f"Filtering samples. Total sample size before filter: {all_samples.size}")

for cfg in cfgs:
    # Unpack relevant config params
    if cfg["build_concept_graph"] is False:
        continue

    labels = cfg["concept_params"]["labels"]
    filter_col = cfg["concept_params"]["filter_col"]

    # Remove incomplete or not well defined samples
    for spl in all_samples:

        # Delete sample from list if there are less than `min_num_cells_per_graph` cells of this kind
        cell_count = 0
        for label in labels:
            list_of_cells = so.obs[spl][filter_col] == label
            cell_count += list_of_cells.sum()

        if cell_count <= min_num_cells_per_graph:
            all_samples = np.delete(all_samples, np.where(all_samples == spl))

        # Remove sample if prediction label is np.nan
        if so.spl.loc[spl][prediction_target] is np.nan:
            all_samples = np.delete(all_samples, np.where(all_samples == spl))

        # Remove sample if prediction label is math.nan
        if type(so.spl.loc[spl][prediction_target]) is not str:
            if math.isnan(so.spl.loc[spl][prediction_target]):
                all_samples = np.delete(all_samples, np.where(all_samples == spl))

# Print to log file
print(f"Sample size after: {all_samples.size}")

# Make map from strings to numbers if the prediction target is a column of stings
if not is_numeric_dtype(so.spl[prediction_target]):
    # Get array of stings with unique labels
    keys = so.spl[prediction_target][all_samples].unique()
    numeric_labels = list(range(0, len(keys)))
    map_to_numeric = dict(zip(keys, numeric_labels))

    # Make new column from so.spl with the prediction label as a numeric value
    prediction_labels = so.spl[prediction_target][all_samples].map(map_to_numeric)

    # Write map to file
    with open(label_dict, "w") as file:
        documents = yaml.dump(map_to_numeric, file, Dumper=yaml.RoundTripDumper)
else:
    prediction_labels = so.spl[prediction_target][all_samples]
    map_to_numeric = {"no_need_for_map": "labels are numeric."}
    with open(label_dict, "w") as file:
        documents = yaml.dump(map_to_numeric, file, Dumper=yaml.RoundTripDumper)

# Save list of samples and their associated prediction labels.
prediction_labels.to_csv(filtered_sample_ids_and_labels, index=True)

# Print to log file
print(f"List of samples saved to: {filtered_sample_ids_and_labels}")
