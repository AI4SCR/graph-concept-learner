#!/usr/bin/env python3

import mlflow
import pandas as pd
import os
from itertools import product
import sys

# Set varaibel names and paths
path_to_all_cfgs = sys.argv[1]
path_to_all_concepts = sys.argv[2]
pred_target = sys.argv[3]
folder_name = sys.argv[4]
split_strategy = sys.argv[5]
root = sys.argv[6]
metric_name = sys.argv[7]
path_dupl = sys.argv[8]
path_miss = sys.argv[9]
path_unf = sys.argv[10]
# path_to_all_cfgs = f"/dccstor/cpath_data/datasets/GCL/{dataset_name}/prediction_tasks/{folder_name}/configs/pretrain_model_configs/"
# path_to_all_concepts = f"/dccstor/cpath_data/datasets/GCL/{dataset_name}/prediction_tasks/{folder_name}/configs/concept_configs/"
# pred_target = "ERStatus"
# dataset_name = "jakson"
# folder_name = "ERStatus_normalized"
# metric_name = "balanced_accuracy"

dataset_name = os.path.basename(root)
name_of_file = f"best_val_{metric_name}.pt"
# Check which experiments have not be run yet
# Get config ids
CFG_IDS = [
    os.path.splitext(f)[0]
    for f in os.listdir(path_to_all_cfgs)
    if os.path.splitext(f)[1] == ".yaml"
]

# Get concept names
CONCEPT_NAMES = [
    os.path.splitext(f)[0]
    for f in os.listdir(path_to_all_concepts)
    if os.path.splitext(f)[1] == ".yaml"
]

# Load all active runs in the experimet
experiment_name = f"san_{dataset_name}_{pred_target}"
l = []

for concept in CONCEPT_NAMES:
    query = f"params.run_type = 'pretrain_concept' and params.concept = '{concept}' and params.folder_name = '{folder_name}' and params.split_strategy = '{split_strategy}'"
    df = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=query,
    )
    l.append(df)

df = pd.concat(l)
print(f"Shape of df is: {df.shape}")

# Check for duplicates
if len(df[df.duplicated(subset=["params.cfg_id", "params.concept"])]) > 0:
    # path_dupl = "./duplicated_runs_cofig_ids.txt"
    print(f"Multiple runs on the same config. Config ids saved to {path_dupl}.")
    duplicated = df[df.duplicated(subset=["params.cfg_id", "params.concept"])][
        "params.path_output_models"
    ]
    duplicated = duplicated + f"/{name_of_file}"
    duplicated.to_csv(path_dupl, header=None, index=None, sep=" ", mode="a")
else:
    print("No duplicate runs found.")

# Get all combinations
workflow_set = set(product(CONCEPT_NAMES, CFG_IDS))

# Get mlflow config concept combinations
mlflow_set = set(df[["params.concept", "params.cfg_id"]].apply(tuple, 1))

# comput set difference
missing_set = workflow_set.difference(mlflow_set)

if len(missing_set) > 0:
    # path_miss = "./missing_runs_cofig_ids.txt"
    print(f"Missing runs found. Missing output files paths saved to {path_miss}.")
    with open(path_miss, "w") as f:
        for concpet_cfg_id in missing_set:
            concept, cfg_id = concpet_cfg_id
            line = f"/dccstor/cpath_data/datasets/GCL/{dataset_name}/prediction_tasks/{pred_target}/{folder_name}/{split_strategy}/checkpoints/{concept}/{cfg_id}/{name_of_file}"
            f.write(f"{line}\n")
else:
    print(
        f"No missing runs. All configs in model_configs/ are present in experiment {experiment_name}."
    )

# Add UNFINISHED runs
unfinished_runs = df[df["status"] != "FINISHED"]["params.path_output_models"]

if len(unfinished_runs) > 0:
    # path_unf = "./unfinished_runs_cofig_ids.txt"
    print(
        f"Not all runs have FINISHED status. UNFINISHED config ids saved to {path_unf}."
    )
    unfinished_runs = unfinished_runs + f"/{name_of_file}"
    unfinished_runs.to_csv(path_unf, header=None, index=None, sep=" ", mode="a")
else:
    print("No unfinished runs found.")
