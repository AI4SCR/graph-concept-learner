#!/usr/bin/env python3
# %%
from tqdm import tqdm
from graph_cl.datasets.concept_set_dataset import ConceptSetDataset
from graph_cl.datasets.concept_dataset import Concept_Dataset
from graph_cl.models.gnn import GNN_plus_MPL
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn as nn
import itertools
import os
import sys
from ruamel import yaml
import mlflow
from torch_geometric import seed_everything
from graph_cl.models.graph_concept_learner import GraphConceptLearner
from graph_cl.utils.mlflow_utils import robust_mlflow, start_mlflow_run
from graph_cl.utils.train_utils import (
    split_concept_dataset,
    get_optimizer_class,
    build_scheduler,
    test_and_log_best_models,
    train_validate_and_log_n_epochs,
    get_dict_of_metric_names_and_paths,
    randomize_labels,
    get_datum_from_ConceptSetDataset,
)

# ### Debug input ###
# paths_to_single_concept_model_configs = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/pretrain_results/best_model_per_concept/best_model_per_concept_ER.yaml"  # Path to file with all the configs.
# path_to_aggregator_and_training_config = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/configs/train_model_configs/57bebcb0-7fde-4e15-ab17-02cd4182acb0.yaml"  # Path to config file
# path_to_datasets = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/processed_data"
# splits_df = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/meta_data/samples_splits.csv"  # Path to df with data splits
# folder_name = "normalized_with_min_max"  # Name of folder
# split_strategy = "split_basel_leave_zurich_as_external"
# pred_target = "ERStatus"  # Prediction target
# root = "/dccstor/cpath_data/datasets/GCL/jakson"  # Path to the dir with all the data (used to specify mlflow experiment)
# log_frequency = 1
# out_file_1 = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/checkpoints/graph_concept_learners/ER/57bebcb0-7fde-4e15-ab17-02cd4182acb0/best_val_balanced_accuracy.pt"  # Paths to the output file with the final model
# out_file_2 = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/checkpoints/graph_concept_learners/ER/57bebcb0-7fde-4e15-ab17-02cd4182acb0/best_val_weighted_f1_score.pt"
# run_type = "train_ER_gcl"

# For local run
# exclude_this_concepts = [
#     "all_cells_contact",
#     "all_cells_radius",
#     "all_cells_ERless_contact",
#     "all_cells_ERless_radius",
#     "endothelial_ERless_contact",
#     "endothelial_stromal_ERless_contact",
#     "endothelial_tumor_ERless_contact",
#     "immune_ERless_radius",
#     "immune_endothelial_ERless_radius",
#     "immune_stromal_ERless_radius",
#     "immune_tumor_ERless_radius",
#     "stromal_ERless_contact",
#     "stromal_tumor_ERless_contact",
#     "tumor_ERless_contact",
# ]
# randomize = "True"
# pred_target = "ERStatus"
# path_to_datasets="/Users/ast/Documents/GitHub/datasets/jakson/prediction_tasks/ERStatus/processed_data"
# randomize="True"
# splits_df="/Users/ast/Downloads/sample_splits.csv"

### Read input and output ###
paths_to_single_concept_model_configs = sys.argv[
    1
]  # Path to file with all the configs.
path_to_aggregator_and_training_config = sys.argv[2]  # Path to config file
path_to_datasets = sys.argv[3]  # Path to concept datasets
splits_df = sys.argv[4]  # Path to df with data splits
folder_name = sys.argv[5]  # Name of folder
split_strategy = sys.argv[6]
randomize = sys.argv[7]
pred_target = sys.argv[8]  # Prediction target
root = sys.argv[
    9
]  # Path to the dir with all the data (used to specify mlflow experiment)
log_frequency = sys.argv[10]
out_file_1 = sys.argv[11]  # Paths to the output file with the final model
out_file_2 = sys.argv[12]
run_type = sys.argv[13]
exclude_this_concepts = sys.argv[14:]  # Exclude the follwoing concepts

#### Load configs ###
# GCL dedicated config
with open(path_to_aggregator_and_training_config) as file:
    gcl_cfg = yaml.load(file, Loader=yaml.Loader)

# individuals GNN dedicated config with paths to other GNN configs
with open(paths_to_single_concept_model_configs) as file:
    cfg_single_concept_models = yaml.load(file, Loader=yaml.Loader)

### Load dataset ###
dataset = ConceptSetDataset(root=path_to_datasets, exclude=exclude_this_concepts)

# Get sample ids for each split
# Get a separate dataset for each split
concept_set_splited_dataset = split_concept_dataset(
    splits_df=splits_df, index_col="core", dataset=dataset
)

# Set seed
seed_everything(gcl_cfg["seed"])

# Permute labels if randomize is "true"
if randomize == "True":
    concept_set_splited_dataset = randomize_labels(
        splits_df, pred_target, concept_set_splited_dataset
    )

# Set torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure the loader keeps track of the concept wise batches
feat = ["x", "edge_index"]

follow_this = []
for i, j in itertools.product(dataset.concepts_names, feat):
    follow_this.append(f"{i}__{j}")

follow_this.append("y")

# Load datasets according to device
loaders = {}
for split_and_splited_dataset in concept_set_splited_dataset.items():
    # Unpack key and value
    split, splited_dataset = split_and_splited_dataset

    # Build loader from (in-memory) list of datums
    loaders[split] = DataLoader(
        [
            get_datum_from_ConceptSetDataset(randomize, device, idx, splited_dataset)
            for idx in range(len(splited_dataset))
        ],
        batch_size=gcl_cfg["batch_size"],
        shuffle=True,
        follow_batch=follow_this,
    )

### Load concept GNN models ###
# Model dictionary
model_dict = {}

# Load models
for concept, path_to_model_config_and_checkpoint in cfg_single_concept_models.items():
    # Unpack paths
    path_to_model_config, path_to_model_checkpoint = path_to_model_config_and_checkpoint
    #
    # Since we are already looping this dict save it to the gcl_cfg s.t. it is logged to mlflow
    gcl_cfg[f"{concept}.path_to_gnn_config"] = path_to_model_config
    gcl_cfg[f"{concept}.path_to_gnn_checkpoint"] = path_to_model_checkpoint
    #
    # Load it
    with open(path_to_model_config) as file:
        concept_cfg = yaml.load(file, Loader=yaml.Loader)
    #
    # Load dataset
    concept_dataset = Concept_Dataset(dataset.concept_dict[concept])
    #
    # Get concept training dataset (needed it to instantiate PNA GNN models)
    concept_splited_dataset = split_concept_dataset(
        splits_df=splits_df, index_col="core", dataset=concept_dataset
    )
    #
    # Get number of clases
    concept_cfg["num_classes"] = concept_dataset.num_classes
    concept_cfg["in_channels"] = concept_dataset.num_node_features
    concept_cfg["hidden_channels"] = concept_cfg["in_channels"] * concept_cfg["scaler"]
    #
    # Load model
    model = GNN_plus_MPL(concept_cfg, concept_splited_dataset["train"])
    #
    # Load checkpoint
    if "end_to_end" not in gcl_cfg.keys():
        model.load_state_dict(torch.load(path_to_model_checkpoint, map_location=device))
    elif gcl_cfg["end_to_end"]:
        pass
    else:
        model.load_state_dict(torch.load(path_to_model_checkpoint, map_location=device))
    #
    # Remove head
    model = model.get_submodule("gnn")
    #
    # Add to dictionary
    model_dict[concept] = model

### Check if all models have the same output dimension ###
out_dims = np.array([])
for concept, model in model_dict.items():
    out_dims = np.append(out_dims, model.gnn.out_channels)
assert all(
    out_dims == out_dims[0]
), f"Not all graph embeddings for the different concept learners are the same dimension."

### Compleate config ###
# Save embedding size to variable
gcl_cfg["emb_size"] = int(out_dims[0])
gcl_cfg["num_classes"] = concept_dataset.num_classes
gcl_cfg["num_concepts"] = dataset.num_concepts

### Instatiate full model. Concept GNN plus agreggator ###
graph_concept_learner = GraphConceptLearner(
    concept_learners=nn.ModuleDict(model_dict),
    config=gcl_cfg,
    device=device,
)

# Move model to device
graph_concept_learner.to(device)

### Define optimizer loss and lr shceduler ###
# Define optimizer
optimizer_class = get_optimizer_class(gcl_cfg)

# If the gnns_lr = 0 the freeze parameters in model
if gcl_cfg["gnns_lr"] == 0:
    for parameter in graph_concept_learner.concept_learners.parameters():
        parameter.requires_grad = False
    optimizer = optimizer_class(
        graph_concept_learner.parameters(), lr=gcl_cfg["agg_lr"]
    )
else:
    # Initialize optimizer with different lrs for the aggregator and gnns
    optimizer = optimizer_class(
        [
            {
                "params": graph_concept_learner.concept_learners.parameters(),
                "lr": gcl_cfg["gnns_lr"],
            },
            {"params": graph_concept_learner.aggregator.parameters()},
        ],
        lr=gcl_cfg["agg_lr"],
    )

# Define loss function.
criterion = torch.nn.CrossEntropyLoss()

# Define learning rate decay strategy
scheduler = build_scheduler(gcl_cfg, optimizer)

### Define mlflow experiment ###
# Make out_dir if it does not already exist
out_dir = os.path.dirname(out_file_1)
os.makedirs(out_dir, exist_ok=True)
start_mlflow_run(root, pred_target, out_dir)

# Add additional information to config s.t. it is logged
cfg_file_name = os.path.basename(path_to_aggregator_and_training_config)
cfg_id = os.path.splitext(cfg_file_name)[0]
gcl_cfg["run_type"] = run_type
gcl_cfg["folder_name"] = folder_name
gcl_cfg["split_strategy"] = split_strategy
gcl_cfg["cfg_id"] = cfg_id
gcl_cfg["path_input_config"] = path_to_aggregator_and_training_config
gcl_cfg["path_output_models"] = out_dir
gcl_cfg.pop("mlp_act")

# Log config
robust_mlflow(mlflow.log_params, params=gcl_cfg)

### Training and evaluation ###
# Log frequency in terms of epochs
log_every_n_epochs = int(log_frequency)

# Save checkpoints for the follwoing metrics
follow_this_metrics = get_dict_of_metric_names_and_paths(out_file_1, out_file_2)

# Train and validate for cfg["n_epochs"]
train_validate_and_log_n_epochs(
    cfg=gcl_cfg,
    model=graph_concept_learner,
    train_loader=loaders["train"],
    val_loader=loaders["val"],
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    log_every_n_epochs=log_every_n_epochs,
    device=device,
    follow_this_metrics=follow_this_metrics,
)

### Load best models an compute test metrics ###
test_and_log_best_models(
    cfg=gcl_cfg,
    model=graph_concept_learner,
    test_loader=loaders["test"],
    criterion=criterion,
    device=device,
    follow_this_metrics=follow_this_metrics,
    out_dir=out_dir,
    split="test",
)

# Test external_test if present
if "external_test" in loaders.keys():
    test_and_log_best_models(
        cfg=gcl_cfg,
        model=graph_concept_learner,
        test_loader=loaders["external_test"],
        criterion=criterion,
        device=device,
        follow_this_metrics=follow_this_metrics,
        out_dir=out_dir,
        split="external_test",
    )

# End run
mlflow.end_run()
