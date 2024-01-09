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
from torch_geometric import seed_everything
from graph_cl.models.graph_concept_learner import GraphConceptLearner
from graph_cl.utils.train_utils import (
    split_concept_dataset,
    get_datum_from_ConceptSetDataset,
)

### Debug input ###
paths_to_single_concept_model_configs = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/pretrain_results/best_model_per_concept/best_model_per_concept_ERless.yaml"  # Path to file with all the configs.
path_to_aggregator_and_training_config = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/configs/best_gcl_ERless_transformer/f6ae460f-ddbc-4d93-9be0-83dc6079b015.yaml"  # Path to config file
path_to_checkpoint = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/checkpoints/graph_concept_learners/best_gcl_ERless_transformer/f6ae460f-ddbc-4d93-9be0-83dc6079b015/best_val_balanced_accuracy.pt"
path_to_datasets = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/processed_data"
splits_df = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/meta_data/samples_splits.csv"  # Path to df with data splits
exclude_this_concepts = [
    "all_cells_contact",
    "all_cells_radius",
    "all_cells_ERless_contact",
    "all_cells_ERless_radius",
    "endothelial_contact",
    "endothelial_stromal_contact",
    "endothelial_tumor_contact",
    "immune_radius",
    "immune_endothelial_radius",
    "immune_stromal_radius",
    "immune_tumor_radius",
    "stromal_contact",
    "stromal_tumor_contact",
    "tumor_contact",
]
randomize = "False"
path_to_out_dir = "/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/train_results/attention_weights"

#### Load configs ###
# GCL dedicated config
with open(path_to_aggregator_and_training_config) as file:
    gcl_cfg = yaml.load(file, Loader=yaml.Loader)

# individuals GNN dedicated config with paths to other GNN configs
with open(paths_to_single_concept_model_configs) as file:
    cfg_single_concept_models = yaml.load(file, Loader=yaml.Loader)

### Load dataset ###
dataset = ConceptSetDataset(root=path_to_datasets, exclude=exclude_this_concepts)

# Set seed
seed_everything(gcl_cfg["seed"])

# Set torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure the loader keeps track of the concept wise batches
feat = ["x", "edge_index"]

follow_this = []
for i, j in itertools.product(dataset.concepts_names, feat):
    follow_this.append(f"{i}__{j}")

follow_this.append("y")

# Load data
loader = DataLoader(
    [
        get_datum_from_ConceptSetDataset(randomize, device, idx, dataset)
        for idx in range(len(dataset))
    ],
    batch_size=1,
    shuffle=False,
    follow_batch=follow_this,
)

### Load concept GNN models ###
# Model dictionary
model_dict = {}

# Load gnn models.
for concept, path_to_model_config_and_checkpoint in cfg_single_concept_models.items():
    # Unpack paths
    # path_to_model_checkpoint not used in this script
    path_to_model_config, path_to_model_checkpoint = path_to_model_config_and_checkpoint
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
gcl_cfg["need_weights"] = True

### Instatiate full model. Concept GNN plus agreggator ###
graph_concept_learner = GraphConceptLearner(
    concept_learners=nn.ModuleDict(model_dict),
    config=gcl_cfg,
    device=device,
)

# Move model to device
graph_concept_learner.load_state_dict(
    torch.load(path_to_checkpoint, map_location=device)
)
graph_concept_learner.to(device)

# Freeze parameters
graph_concept_learner.eval()

# Go through loader and save attention weights.
true_positives = {}
true_negatives = {}
false_positives = {}
false_negatives = {}

for data in loader:
    # Unpack output
    unnorm_logits, attention_weights = graph_concept_learner(data)
    #
    # Take prediction
    y_pred = unnorm_logits.argmax(dim=1)
    #
    # Categorize attention map
    # If predicted as positive and true label is positive
    if y_pred == torch.tensor(0, device=device) and data.y == torch.tensor(
        [0], device=device
    ):
        true_positives[data.sample_id[0]] = (
            torch.squeeze(attention_weights).cpu().detach()
        )
    elif y_pred == torch.tensor(1, device=device) and data.y == torch.tensor(
        [0], device=device
    ):
        false_negatives[data.sample_id[0]] = (
            torch.squeeze(attention_weights).cpu().detach()
        )
    elif y_pred == torch.tensor(1, device=device) and data.y == torch.tensor(
        [1], device=device
    ):
        true_negatives[data.sample_id[0]] = (
            torch.squeeze(attention_weights).cpu().detach()
        )
    elif y_pred == torch.tensor(0, device=device) and data.y == torch.tensor(
        [1], device=device
    ):
        false_positives[data.sample_id[0]] = (
            torch.squeeze(attention_weights).cpu().detach()
        )

# print(f"true_positives: {len(true_positives)}")
# print(f"true_negatives: {len(true_negatives)}")
# print(f"false_positives: {len(false_positives)}")
# print(f"false_negatives: {len(false_negatives)}")
torch.save(
    true_positives, os.path.join(path_to_out_dir, "true_positives_tensor_dict.pt")
)
torch.save(
    true_negatives, os.path.join(path_to_out_dir, "true_negatives_tensor_dict.pt")
)
torch.save(
    false_positives, os.path.join(path_to_out_dir, "false_positives_tensor_dict.pt")
)
torch.save(
    false_negatives, os.path.join(path_to_out_dir, "false_negatives_tensor_dict.pt")
)

# true_positives = 0
# true_negatives = 0
# false_positives = 0
# false_negatives = 0

# for data in tqdm(loader):
#     # Unpack output
#     unnorm_logits = graph_concept_learner(data)
#     #
#     # Take prediction
#     y_preds = unnorm_logits.argmax(dim=1)
#     #
#     for y_pred, y_true in zip(y_preds,data.y):
#         # Categorize attention map
#         # If predicted as positive and true label is positive
#         if y_pred == torch.tensor(0, device=device) and y_true == torch.tensor(
#             [0], device=device
#         ):
#             true_positives += 1
#         elif y_pred == torch.tensor(1, device=device) and y_true == torch.tensor(
#             [0], device=device
#         ):
#             false_negatives += 1
#         elif y_pred == torch.tensor(1, device=device) and y_true == torch.tensor(
#             [1], device=device
#         ):
#             true_negatives += 1
#         elif y_pred == torch.tensor(0, device=device) and y_true == torch.tensor(
#             [1], device=device
#         ):
#             false_positives += 1


# print(f"true_positives: {true_positives}")
# print(f"true_negatives: {true_negatives}")
# print(f"false_positives: {false_positives}")
# print(f"false_negatives: {false_negatives}")
