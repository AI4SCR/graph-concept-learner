# %%
from graph_cl.datasets.concept_dataset import Concept_Dataset
import os.path as osp
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.config import assert_cfg
from torch_geometric.graphgym.train import train
from yacs.config import CfgNode as CN
import torch
import argparse
import os

# %% Parse command line arguments to pass to the script
parser = argparse.ArgumentParser(
    description="Concept-Graph model pre-trainig.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("cfg", help="Path to model config file.")
args = parser.parse_args()
args = vars(args)

# %% Load config and model
## Instantiate config object
cfg = CN()
# cfg_path = osp.join(args["cfg"])
cfg_path = "/Users/ast/Documents/GitHub/pytorch_geometric/graphgym/results/base_cl_grid_concept_grid/base_cl-dataset=immune_tumor_radius-layer=gcnconv/config.yaml"
with open(cfg_path, "r") as f:
    cfg = cfg.load_cfg(f)

assert_cfg(cfg)

# %% Set seed
seed_everything(cfg.seed)

# %% Instantiate concept dataset
dataset = Concept_Dataset(
    osp.join(cfg.dataset.dir, "concepts", cfg.dataset.name, "data")
)
# dataset = Concept_Dataset(
#     root="/Users/ast/Documents/GitHub/datasets/clinical_type/concepts/immune_tumor_contact/data"
# )

# Shouffle and split
dataset = dataset.shuffle()
split_indx = int(dataset.len() * cfg.dataset.split[0])
# split_indx = int(dataset.len() * 0.8)
train_dataset = dataset[:split_indx]
test_dataset = dataset[split_indx:]

# Load
train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

# Print epoch
# for step, data in enumerate(train_loader):
#     print(f"Step {step + 1}:")
#     print("=======")
#     print(f"Number of graphs in the current batch: {data.num_graphs}")
#     print(data)
#     print()

## Assert
cfg.share.dim_in = dataset[0].x.shape[1]
cfg.share.dim_out = dataset.num_classes
assert cfg.share.dim_in == dataset[0].x.shape[1]
assert cfg.share.dim_out == dataset.num_classes

## Build model
model = GNN(cfg.share.dim_in, cfg.share.dim_out, cfg)

# %% Print model info
model_name = "Example name"
print(f"Model name: {model_name}", end="\n\n")
print("Architecture:", end="\n\n")
print(model, end="\n\n")
print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}", end="\n\n")

# %% Define train and test functions
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr)
criterion = torch.nn.CrossEntropyLoss()

# Define train and test functions
def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out, y = model(data)  # Perform a single forward pass.
        loss = criterion(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out, y = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


# %% Train and test
for epoch in range(1, cfg.optim.max_epoch):
    train()
    if epoch % 10 == 0:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

# Make dir
model_path = osp.join(cfg.dataset.dir, "concepts", cfg.dataset.name, "model")
# model_path = "/Users/ast/Documents/GitHub/datasets/clinical_type/concepts/immune_tumor_contact/model/example_model"
os.makedirs(model_path, exist_ok=True)

# Save model to file
# file_name = osp.join(args["model_dir"], "model_params.pt")
file_name = osp.join(model_path, "model_params.pt")
torch.save(model.state_dict(), file_name)

print()
print("Done!")

# %%
