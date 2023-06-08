#!/usr/bin/env python3
from graph_cl.models.space_gm_mlp import MLP_pred
from graph_cl.utils.train_utils import get_dict_of_metric_names_and_paths
from graph_cl.utils.mlflow_utils import robust_mlflow, start_mlflow_run
from torch_geometric import seed_everything
import pandas as pd
import numpy as np
import os
import mlflow
import torch
from ruamel import yaml
import sys

# Read CL args
folder_name = sys.argv[1]
split_strategy = sys.argv[2]
pred_target = sys.argv[3]
root = sys.argv[4]
path_to_cfg = sys.argv[5]
path_to_splits = sys.argv[6]
path_to_data = sys.argv[7]
log_frequency = sys.argv[8]
out_file_1 = sys.argv[9]
out_file_2 = sys.argv[10]

# root="/Users/ast/Documents/GitHub/datasets/jakson"
# pred_target="ERStatus"
# out_file_1="/Users/ast/Downloads/best_val_weighted_f1_score.pt"
# out_file_2="/Users/ast/Downloads/best_val_weighted_f1_score.pt"
# folder_name="ERStatus_norm"
# log_frequency = 1
# path_to_cfg = "/Users/ast/Documents/GitHub/datasets/jakson/prediction_targets/ERStatus/configs/space_gm_configs/cfg_id.yaml"
# path_to_splits="/Users/ast/Documents/GitHub/datasets/jakson/prediction_tasks/ERStatus/meta_data/samples_splits.csv"
# path_to_data="/Users/ast/Documents/GitHub/datasets/jakson/prediction_targets/ERStatus/non_spatial_baseline/data/composition_vectors.csv"
# # # path_to_data="/Users/ast/Documents/GitHub/datasets/jakson/prediction_targets/ERStatus/non_spatial_baseline/data/composition_vectors_norm.csv"
# cfg = {
#     "n_mlp_layers":3,
#     "scalar":1,
#     "batch_size":4,
#     "n_epochs":200,
#     "lr":0.001,
#     "seed":1,
#     "scheduler":["LambdaLR", 0.5, 20]
# }

# Read config
with open(path_to_cfg) as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

# Read data
split_map = pd.read_csv(path_to_splits, index_col="core")
all_data = pd.read_csv(path_to_data, index_col=0)

# Init dictionary of subseted datasets
datasets = {}

# Loop over splits
for split in split_map["split"].unique():
    ids = split_map[split_map["split"] == split].index.values
    X_and_y = all_data[all_data.index.isin(ids)]
    y = X_and_y["y"].to_numpy()
    X = X_and_y.drop("y", axis=1).to_numpy()
    datasets[split] = {"X": X, "y": y}

# Instantiate model
seed_everything(cfg["seed"])
model = MLP_pred(
    n_feat=datasets["train"]["X"].shape[1],
    n_layers=cfg["n_mlp_layers"],
    scalar=cfg["scalar"],
    n_tasks=len(np.unique(all_data["y"].values)),
    gpu=torch.cuda.is_available(),
    task="classification",
    balanced=True,
)

# Make out_dir if it does not already exist
out_dir = os.path.dirname(out_file_1)
os.makedirs(out_dir, exist_ok=True)

# Start mlflow experiment
start_mlflow_run(root, pred_target, out_dir)
cfg_file_name = os.path.basename(path_to_cfg)
cfg["cfg_id"] = os.path.splitext(cfg_file_name)[0]
cfg["run_type"] = "space_gm"
data_dir_path = os.path.dirname(path_to_data)
cfg["dataset_name"] = os.path.basename(data_dir_path)
cfg["folder_name"] = folder_name
cfg["split_strategy"] = split_strategy
cfg["path_input_config"] = path_to_cfg
cfg["path_output_models"] = out_dir

# Log config
robust_mlflow(mlflow.log_params, params=cfg)

# Save checkpoints for the follwoing metrics
follow_this_metrics = get_dict_of_metric_names_and_paths(out_file_1, out_file_2)

# Train model for n_epochs and log performance to mlflow
model.fit(
    X=datasets["train"]["X"],
    y=datasets["train"]["y"],
    X_val=datasets["val"]["X"],
    y_val=datasets["val"]["y"],
    follow_this_metrics=follow_this_metrics,
    cfg=cfg,
    batch_size=cfg["batch_size"],
    n_epochs=cfg["n_epochs"],
    lr=cfg["lr"],
    log_every_n_epochs=int(log_frequency),
)

# Evaluate on test split
model.test_and_log_best_models(
    split="test",
    X=datasets["test"]["X"],
    y=datasets["test"]["y"],
    follow_this_metrics=follow_this_metrics,
    out_dir=out_dir,
)

# Evaluatre on external test set
if "external_test" in datasets.keys():
    model.test_and_log_best_models(
        split="external_test",
        X=datasets["external_test"]["X"],
        y=datasets["external_test"]["y"],
        follow_this_metrics=follow_this_metrics,
        out_dir=out_dir,
    )

# End run
mlflow.end_run()
