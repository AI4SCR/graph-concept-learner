#!/usr/bin/env python3

import mlflow
import os
import sys
import pandas as pd
from ruamel import yaml

(path_to_all_concepts, folder_name, pred_target, root, out_file) = sys.argv

# Get concept names
CONCEPT_NAMES = [
    os.path.splitext(f)[0]
    for f in os.listdir(path_to_all_concepts)
    if os.path.splitext(f)[1] == ".yaml"
]

# Define exp name
dataset_name = os.path.basename(root)
experiment_name = f"san_{dataset_name}_{pred_target}"

# Specify info to save and how to order the table
save_cols = [
    "run_id",
    "metrics.test_best_val_balanced_accuracy_balanced_accuracy",
    "metrics.test_best_val_weighted_f1_score_balanced_accuracy",
    "metrics.test_best_val_balanced_accuracy_weighted_f1_score",
    "metrics.test_best_val_weighted_f1_score_weighted_f1_score",
    "params.path_input_config",
    "params.path_output_models",
    "params.hidden_channels",
]

metric_base_name = "best_val_balanced_accuracy"
metric_name = f"test_{metric_base_name}_balanced_accuracy"
sort_by = f"metrics.{metric_name}"

list_of_hidden_channels = []
df_dict = {}

# Get a df for every concept
for concept in CONCEPT_NAMES:
    # Get df
    df = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"params.concept = '{concept}' and params.folder_name = '{folder_name}'",
    )

    # Select columns of interest and sort
    df = df[save_cols].sort_values(by=sort_by, ascending=False)

    # Save to dict
    df_dict[concept] = df

    # Save the hiddent dimention of the best performing model
    # TODO: check what this value is
    list_of_hidden_channels.append(df["params.hidden_channels"].iloc[0])

unique_hidden_cahnnels = set(list_of_hidden_channels)

max_count = 0
most_common_channel = ""

for unique_cahnnel in unique_hidden_cahnnels:
    if list_of_hidden_channels.count(unique_cahnnel) > max_count:
        most_common_channel = unique_cahnnel

# Get a df for every concept
out_dict = {}
for concept in CONCEPT_NAMES:
    paths = df.loc[
        df["params.hidden_channels"] == most_common_channel,
        ["params.path_input_config", "params.path_output_models"],
    ].iloc[0]
    paths[1] = os.path.join(paths[1], f"{metric_base_name}.pt")
    out_dict[concept] = list(paths)

# Write to file
with open(out_file, "w") as file:
    yaml.dump(out_dict, file, Dumper=yaml.RoundTripDumper)
