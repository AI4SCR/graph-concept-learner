#!/usr/bin/env python3

import yaml
import pickle

# import numpy as np
import pandas as pd

# import math
from pandas.api.types import is_numeric_dtype
from ..configuration import CONFIG


def filter():
    # Cast to int
    min_num_cells_per_graph = CONFIG.filter.min_num_cells_per_graph
    cfgs_dir = CONFIG.experiment.root / "configuration" / "concept_configs"
    so_file = CONFIG.data.intermediate
    prediction_target = CONFIG.experiment.prediction_target
    filtered_sample_ids_and_labels = (
        CONFIG.experiment.root / "filter" / "filtered_sample_ids_and_labels.csv"
    )
    filtered_sample_ids_and_labels.parent.mkdir(parents=True, exist_ok=True)
    label_dict = CONFIG.experiment.root / "filter" / "label_dict.yaml"
    label_dict.parent.mkdir(parents=True, exist_ok=True)

    # Load all configs (one for each concept)
    cfgs = []
    for p in cfgs_dir.glob("*.yaml"):

        with open(p) as f:
            cfg = yaml.safe_load(f)

        # append to list
        cfgs.append(cfg)

    # %% Load so object
    with open(so_file, "rb") as f:
        so = pickle.load(f)

    # List all the samples
    all_samples = set(so.spl.index)

    # Print log file sample size before filtering
    print(f"Filtering samples. Total sample size before filter: {len(all_samples)}")

    for cfg in cfgs:
        # Unpack relevant config params
        if cfg["build_concept_graph"] is False:
            continue

        labels = cfg["concept_params"]["include_labels"]
        filter_col = cfg["concept_params"]["filter_col"]

        # Remove incomplete or not well defined samples
        sample_to_remove = set()
        for spl in all_samples:

            # Delete sample from list if there are less than `min_num_cells_per_graph` cells of this kind
            # cell_count = 0
            # for label in labels:
            #     list_of_cells = so.obs[spl][filter_col] == label
            #     cell_count += list_of_cells.sum()
            cell_count = so.obs[spl][filter_col].isin(labels).sum()

            if cell_count <= min_num_cells_per_graph or pd.isna(
                so.spl.loc[spl][prediction_target]
            ):
                # all_samples = np.delete(all_samples, np.where(all_samples == spl))
                sample_to_remove.add(spl)

            # # Remove sample if prediction label is np.nan
            # if so.spl.loc[spl][prediction_target] is np.nan:
            #     # all_samples = np.delete(all_samples, np.where(all_samples == spl))
            #     all_samples.remove(spl)

            # Remove sample if prediction label is math.nan
            # if type(so.spl.loc[spl][prediction_target]) is not str:
            #     if math.isnan(so.spl.loc[spl][prediction_target]):
            #         all_samples = np.delete(all_samples, np.where(all_samples == spl))

    all_samples = all_samples - sample_to_remove
    # Print to log file
    print(f"Sample size after: {len(all_samples)}")

    # Make map from strings to numbers if the prediction target is a column of stings
    all_samples = list(all_samples)
    if not is_numeric_dtype(so.spl[prediction_target]):
        # Get array of stings with unique labels
        keys = so.spl[prediction_target][all_samples].unique()
        numeric_labels = list(range(0, len(keys)))
        map_to_numeric = dict(zip(keys, numeric_labels))

        # Make new column from so.spl with the prediction label as a numeric value
        prediction_labels = so.spl[prediction_target][all_samples].map(map_to_numeric)

        # Write map to file
        with open(label_dict, "w") as file:
            documents = yaml.dump(map_to_numeric, file)
    else:
        prediction_labels = so.spl[prediction_target][all_samples]
        map_to_numeric = {"no_need_for_map": "labels are numeric."}
        with open(label_dict, "w") as file:
            documents = yaml.dump(map_to_numeric, file)

    # Save list of samples and their associated prediction labels.
    prediction_labels.to_csv(filtered_sample_ids_and_labels, index=True)

    # Print to log file
    print(f"List of samples saved to: {filtered_sample_ids_and_labels}")
