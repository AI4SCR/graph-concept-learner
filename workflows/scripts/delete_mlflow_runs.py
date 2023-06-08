#!/usr/bin/env python3
# %%
import mlflow
import pandas as pd
import os
import sys

# Set varaibel names and paths
path_to_all_concepts = sys.argv[1]
pred_target = sys.argv[2]
folder_name = sys.argv[3]
split_strategy = sys.argv[4]
root = sys.argv[5]
# pred_target = "ERStatus"
# folder_name = "ERStatus_normalized"
# dataset_name = "jakson"
# path_to_all_concepts = f"/dccstor/cpath_data/datasets/GCL/{dataset_name}/prediction_tasks/{folder_name}/configs/dataset_configs/"

# Get datset name
dataset_name = os.path.basename(root)

# String up exp name
experiment_name = f"san_{dataset_name}_{pred_target}"

# Get concept names
CONCEPT_NAMES = [
    os.path.splitext(f)[0]
    for f in os.listdir(path_to_all_concepts)
    if os.path.splitext(f)[1] == ".yaml"
]

# Load all active runs in the experimet
l = []

for concept in CONCEPT_NAMES:
    query = f"params.run_type = 'pretrain_gcl' and params.concept = '{concept}' and params.folder_name = '{folder_name}' and params.split_strategy = '{split_strategy}'"
    df = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=query,
    )
    l.append(df)

# All experiments df
df = pd.concat(l)

# Get runs
run_ids = df["run_id"].values

# %% Check if there are runs in query and if so delete.
if len(run_ids) > 0:
    # Delete runs
    for run_id in run_ids:
        mlflow.delete_run(run_id=run_id)
else:
    print(f"Nothing to delete.")
