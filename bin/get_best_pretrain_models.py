#!/usr/bin/env python3

import mlflow
import pandas as pd
import os
from ruamel import yaml
import sys

### Define settings ###
metric_name = sys.argv[1]
path_to_best_models = sys.argv[2]
path_to_summary = sys.argv[3]
concept_run_ids_path = sys.argv[4]
baselines_run_ids_path = sys.argv[5]

# Read ids
with open(concept_run_ids_path, "r") as f:
    concept_run_ids = f.read().splitlines()

with open(baselines_run_ids_path, "r") as f:
    baselines_run_ids = f.read().splitlines()

# path_to_best_models = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus_normalized/pretrain_results/best_models_per_concept.yaml"
# path_to_summary = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus_normalized/pretrain_results/best_models_per_concept.csv"
# metric_name = "weighted_f1_score"
name_of_file = f"best_val_{metric_name}.pt"
keep_this_cols = [
    "run_id",
    "concept",
    # "test_best_val_weighted_f1_score_balanced_accuracy",
    # "test_best_val_weighted_f1_score_weighted_f1_score",
    # "best_val_weighted_f1_score",
    # "val_balanced_accuracy",
    # 'train_balanced_accuracy',
    # 'train_weighted_precision',
    # 'train_weighted_recall',
    # 'train_weighted_f1_score',
    # 'train_loss',
    # "val_weighted_f1_score",
    "val_loss",
    # 'val_weighted_recall',
    "best_val_balanced_accuracy",
    # 'val_weighted_precision',
    # 'test_best_val_balanced_accuracy_weighted_recall',
    "test_best_val_balanced_accuracy_balanced_accuracy",
    # 'test_best_val_balanced_accuracy_weighted_precision',
    "test_best_val_balanced_accuracy_weighted_f1_score",
    # 'test_best_val_balanced_accuracy_loss',
    # 'test_best_val_weighted_f1_score_weighted_precision',
    # 'test_best_val_weighted_f1_score_weighted_recall',
    # 'test_best_val_weighted_f1_score_loss',
    "gnn",
    "pool",
    # 'in_channels',
    "hidden_channels",
    "num_layers",
    # 'dropout',
    # 'act',
    # 'act_first',
    # 'norm',
    "jk",
    # 'num_layers_MLP',
    "batch_size",
    "lr",
    # 'optim',
    # 'n_epoch',
    "scheduler",
    # 'seed',
    # 'num_classes',
    # 'aggregators',
    "scalers",
    # 'run_type',
    # 'folder_name',
    "cfg_id",
    "path_input_config",
    "path_output_models",
    # 'mlflow.user',
    # 'mlflow.source.name',
    # 'mlflow.source.type',
    # 'mlflow.source.git.commit',
]

### Get run info ###
run_dicts = []

for run_id in concept_run_ids:
    dict_of_dicts = mlflow.get_run(run_id).data.to_dictionary()
    flat_dict = {
        **dict_of_dicts["metrics"],
        **dict_of_dicts["params"],
        **dict_of_dicts["tags"],
    }
    flat_dict["run_id"] = run_id
    run_dicts.append(flat_dict)

df = pd.DataFrame.from_dict(run_dicts)
df = df[keep_this_cols]

### Append model name to path ###
df["path_output_models"] = df["path_output_models"] + f"/{name_of_file}"

# Check if all paths exist
for p in df["path_output_models"].values:
    assert os.path.exists(p), f"File {p} deos not exist!"

for p in df["path_input_config"].values:
    assert os.path.exists(p), f"File {p} deos not exist!"

### Write outputs ###
# Write config for trainer
best_models_dict = {}
for concept in df["concept"].values:
    row = df.loc[df["concept"] == concept]
    best_models_dict[concept] = [
        row.iloc[0]["path_input_config"],
        row.iloc[0]["path_output_models"],
    ]

with open(path_to_best_models, "w") as file:
    yaml.dump(best_models_dict, file, Dumper=yaml.RoundTripDumper)

print(f"Config for trainer in {path_to_best_models}")

### Write table to summary ###
run_dicts = []

for run_id in baselines_run_ids:
    dict_of_dicts = mlflow.get_run(run_id).data.to_dictionary()
    flat_dict = {
        **dict_of_dicts["metrics"],
        **dict_of_dicts["params"],
        **dict_of_dicts["tags"],
    }
    flat_dict["run_id"] = run_id
    run_dicts.append(flat_dict)

dfb = pd.DataFrame.from_dict(run_dicts)
dfb = dfb[keep_this_cols]
df2 = pd.concat([df, dfb], ignore_index=True)
df2.to_csv(path_to_summary)
print(f"Summary in {path_to_summary}")
