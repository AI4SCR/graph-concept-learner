#!/usr/bin/env python3
# from tqdm import tqdm
import torch
import itertools
import os
import sys
import yaml
import mlflow
import torch.nn as nn
import numpy as np
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from graph_cl.datasets.concept_set_dataset import ConceptSetDataset
from graph_cl.datasets.concept_dataset import Concept_Dataset
from graph_cl.models.graph_concept_learner import GraphConceptLearner
from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.utils.mlflow_utils import robust_mlflow, start_mlflow_run
from graph_cl.utils.train_utils import (
    split_concept_dataset,
    get_optimizer_class,
    build_scheduler,
    test_and_log_best_models,
    train_validate_and_log_n_epochs,
    get_dict_of_metric_names_and_paths,
    permute_labels,
)

# Read input and output
(
    program_name,
    paths_to_concept_set,  # Config with paths to file with all the configs.
    path_to_aggregator_and_training_config,  # Path to config file
    splits_df,  # Path to df with data splits
    mlflow_on_remote_server,
    mlflow_uri,
    run_type,
    normalized_with,  # Name of folder
    split_strategy,
    labels_permuted,
    seed,
    pred_target,  # Prediction target
    root,  # Path to the dir with all the data (used to specify mlflow experiment)
    log_frequency,
    out_file_1,  # Paths to the output file with the final model
    out_file_2,
) = sys.argv

# Load configs
# GCL dedicated config
with open(path_to_aggregator_and_training_config) as file:
    gcl_cfg = yaml.load(file, Loader=yaml.Loader)

# Single concept configs with paths to other GNN models, a checkpoint and data
with open(paths_to_concept_set) as file:
    concept_set_cfg = yaml.load(file, Loader=yaml.Loader)

# Completing the path to data with the current split
for key in concept_set_cfg.keys():
    concept_set_cfg[key]["data"] = os.path.join(
        concept_set_cfg[key]["data"], os.path.basename(splits_df).split(".")[0]
    )

# Load dataset
dataset = ConceptSetDataset(config=concept_set_cfg)

# Get sample ids for each split
# Get a separate dataset for each split
concept_set_splitted_dataset = split_concept_dataset(
    splits_df=splits_df, index_col="core", dataset=dataset
)

# Set seed
gcl_cfg["seed"] = int(seed.split("_")[1])
seed_everything(gcl_cfg["seed"])

# Permute labels if labels_permuted is "permuted"
if labels_permuted == "permuted":
    concept_set_splitted_dataset = permute_labels(
        splits_df, pred_target, concept_set_splitted_dataset
    )

# Make sure the loader keeps track of the concept wise batches
follow_this = ["y"]
for i, j in itertools.product(dataset.concept_names, ["x", "edge_index"]):
    follow_this.append(f"{i}__{j}")

# Set torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make loaders for each split
loaders = {}
for split, splitted_dataset in concept_set_splitted_dataset.items():

    # Build loader from (in-memory) list of datums
    loaders[split] = DataLoader(
        [
            splitted_dataset[idx].to(device, non_blocking=True)
            for idx in range(len(splitted_dataset))
        ],
        batch_size=gcl_cfg["batch_size"],
        shuffle=True,
        follow_batch=follow_this,
    )

# Load concept GNN models
model_dict = {}  # Init model dictionary

# Load models
for concept, data_model_checkpoints in concept_set_cfg.items():
    # Unpack paths
    path_to_model_config = concept_set_cfg[concept]["config"]
    path_to_model_checkpoint = concept_set_cfg[concept]["checkpoint"]

    # Since we are already looping this dict save it to the gcl_cfg s.t. it is logged to mlflow
    gcl_cfg[f"{concept}.path_to_gnn_config"] = path_to_model_config
    gcl_cfg[f"{concept}.path_to_gnn_checkpoint"] = path_to_model_checkpoint

    # Load it
    with open(path_to_model_config) as file:
        concept_cfg = yaml.load(file, Loader=yaml.Loader)

    # Load dataset
    concept_dataset = Concept_Dataset(dataset.concept_dict[concept])

    # Get concept training dataset (needed it to instantiate PNA GNN models)
    concept_splitted_dataset = split_concept_dataset(
        splits_df=splits_df, index_col="core", dataset=concept_dataset
    )

    # Get number of classes
    concept_cfg["num_classes"] = concept_dataset.num_classes
    concept_cfg["in_channels"] = concept_dataset.num_node_features
    concept_cfg["hidden_channels"] = concept_cfg["in_channels"] * concept_cfg["scaler"]

    # Load model
    model = GNN_plus_MPL(concept_cfg, concept_splitted_dataset["train"])

    # Load checkpoint
    if "end_to_end" not in gcl_cfg.keys():
        model.load_state_dict(torch.load(path_to_model_checkpoint, map_location=device))
    elif gcl_cfg["end_to_end"]:
        pass
    else:
        model.load_state_dict(torch.load(path_to_model_checkpoint, map_location=device))

    # Remove head
    model = model.get_submodule("gnn")

    # Add to dictionary
    model_dict[concept] = model

# Check if all models have the same output dimension
out_dims = np.array([])
for concept, model in model_dict.items():
    out_dims = np.append(out_dims, model.gnn.out_channels)
assert all(
    out_dims == out_dims[0]
), "Not all graph embeddings for the different concept learners are the same dimension."

# Compleat config
# Save embedding size to variable
gcl_cfg["emb_size"] = int(out_dims[0])
gcl_cfg["num_classes"] = concept_dataset.num_classes
gcl_cfg["num_concepts"] = dataset.num_concepts

# Insatiate full model. Concept GNN plus aggregator
graph_concept_learner = GraphConceptLearner(
    concept_learners=nn.ModuleDict(model_dict),
    config=gcl_cfg,
    device=device,
)

# Move model to device
graph_concept_learner.to(device)

# Define optimizer loss and lr scheduler
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

# Define mlflow experiment
# Make out_dir if it does not already exist
out_dir = os.path.dirname(out_file_1)
os.makedirs(out_dir, exist_ok=True)

# Define mlflow experiment
if mlflow_on_remote_server == "False":
    mlflow.set_tracking_uri(mlflow_uri)

start_mlflow_run(root, pred_target, out_dir)

# Add additional information to config s.t. it is logged
cfg_file_name = os.path.basename(path_to_aggregator_and_training_config)
cfg_id = os.path.splitext(cfg_file_name)[0]
gcl_cfg["run_type"] = run_type
gcl_cfg["normalized_with"] = normalized_with
gcl_cfg["fold"] = os.path.basename(splits_df).split(".")[0]
gcl_cfg["split_strategy"] = split_strategy
gcl_cfg["cfg_id"] = cfg_id
gcl_cfg["attribute_config"] = os.path.basename(os.path.dirname(paths_to_concept_set))
gcl_cfg["concept_set"] = os.path.basename(paths_to_concept_set).split(".")[0]
gcl_cfg["labels_permuted"] = labels_permuted
gcl_cfg["path_input_config"] = path_to_aggregator_and_training_config
gcl_cfg["path_output_models"] = out_dir
gcl_cfg.pop("mlp_act")

# Log config
robust_mlflow(mlflow.log_params, params=gcl_cfg)

# Training and evaluation
# Log frequency in terms of epochs
log_every_n_epochs = int(log_frequency)

# Save checkpoints for the following metrics
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

# Load best models an compute test metrics
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
