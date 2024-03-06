# %%
from graph_cl.datasets.paradigm_dataset import Paradigm_Dataset
from torch_geometric.loader import DataLoader
import argparse
import torch
import torch.nn as nn
import itertools
from yacs.config import CfgNode as CN
import os.path as osp
import os
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric import seed_everything
from graph_cl.models.transformer import ConceptGraphTransformer
from graph_cl.models.gc_learner import GraphConceptLearner

# %% Parse command line arguments to pass to the script
parser = argparse.ArgumentParser(
    description="Concept-Graph-Learning model trainig.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("cfg", help="Path to config file.")
args = parser.parse_args()
args = vars(args)

# %% Instantiate config object
cfg = CN()
cfg_path = osp.join(args["cfg"])
with open(cfg_path, "r") as f:
    cfg = cfg.load_cfg(f)

# %% Set seed
seed_everything(cfg.seed)

# %% Load dataset
# dataset = Paradigm_Dataset(osp.join(cfg.data_dir,"concepts"))
dataset = Paradigm_Dataset(
    root="/Users/ast/Documents/GitHub/datasets/clinical_type/concepts"
)

# %% Shouffle dataset.
dataset = dataset.shuffle()

# Fet split index
# split_indx = int(dataset.len() * args["training_frac"])
split_indx = int(dataset.len() * cfg.split[0])

# Split data
train_dataset = dataset[:split_indx]
test_dataset = dataset[split_indx:]

# %% Make sure the loader keeps track of the concept wise batches
concept_alisas = [f"c{i}" for i in range(dataset.num_concepts)]
feat = ["x", "edge_index"]

follow_this = []
for i, j in itertools.product(concept_alisas, feat):
    follow_this.append(f"{i}_{j}")

follow_this.append("y")

# %% Load dataset into dataloader
train_loader = DataLoader(
    train_dataset, batch_size=cfg.batch_size, shuffle=True, follow_batch=follow_this
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.batch_size, shuffle=False, follow_batch=follow_this
)

# for step, data in enumerate(train_loader):
#     print(f"Step {step + 1}:")
#     print("=======")
#     print(f"Number of graphs in the current batch: {data.num_graphs}")
#     print(data)
#     print()

# %% Load pretrained models
concept_learners = []

for concept_dir in dataset.concept_dirs:
    # Load config
    cfg_path = osp.join(concept_dir, "model", "model_config.yaml")
    concept_cfg = CN()
    with open(cfg_path, "r") as f:
        concept_cfg = concept_cfg.load_cfg(f)

    # Instantiate model and load pretrained parameters
    model_param_path = osp.join(concept_dir, "model", "model_params.pt")
    model = GNN(concept_cfg.share.dim_in, concept_cfg.share.dim_out, concept_cfg)
    model.load_state_dict(torch.load(model_param_path))

    # Remove last layer and replace
    num_layers_of_post_mp = len(model.get_submodule("post_mp.layer_post_mp.model"))

    # Check if there is even layers at all
    if num_layers_of_post_mp > 0:
        index_of_last_layer = num_layers_of_post_mp - 1
        _ = model.pop(index_of_last_layer)

    # Append new layer
    model.append(nn.Linear(concept_cfg.gnn.dim_inner, cfg.emb_size))

    # Append model to a list
    concept_learners.append(model)

# %% Instantiate transformer model
transformer = ConceptGraphTransformer(
    emb_size=cfg.emb_size,
    n_heads=cfg.n_heads,
    depth=cfg.depth,
    n_classes=dataset.num_classes,
    dim_feedforward=cfg.dim_feedforward,
)

# Instantiate concept learner model
graph_concept_learner = GraphConceptLearner(
    concept_learners=concept_learners,
    transformer=transformer,
    emb_size=cfg.emb_size,
    num_concepts=dataset.num_concepts,
    batch_size=cfg.batch_size,
)

# %% Define train and test functions
optimizer = torch.optim.Adam(graph_concept_learner.parameters(), lr=cfg.lr)
criterion = torch.nn.CrossEntropyLoss()

# Define train and test functions
def train():
    graph_concept_learner.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        y = data.y
        out = graph_concept_learner(data)  # Perform a single forward pass.
        loss = criterion(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    graph_concept_learner.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        y = data.y
        out = graph_concept_learner(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


# %% Train and test
for epoch in range(1, cfg.max_epoch):
    train()
    if epoch % 10 == 0:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

# Make dir
model_path = osp.join(cfg.dir, "models", "model_name_example")
# model_path = "/Users/ast/Documents/GitHub/datasets/clinical_type/concepts/immune_tumor_contact/model/example_model"
os.makedirs(model_path, exist_ok=True)

# Save model to file
# file_name = osp.join(args["model_dir"], "model_params.pt")
file_name = osp.join(model_path, "model_params.pt")
torch.save(graph_concept_learner.state_dict(), file_name)

print()
print("Done!")
