import torch
import torch.nn as nn
from torch_geometric import seed_everything
import yaml
import pandas as pd
from pathlib import Path
from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.datasets.ConceptDataset import CptDatasetMemo
import yaml
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.models.graph_concept_learnerV2 import GraphConceptLearner
from graph_cl.train.lightning import LitModule

from graph_cl.datasets.ConceptDataset import CptDatasetMemo
from graph_cl.configuration.configurator import Training

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path
import pandas as pd

# %%

folds_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds"
)
fold_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
)

concept_configs_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/00_concepts"
)

train_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/train.yaml"
)
with open(train_config_path, "r") as f:
    train_config = yaml.safe_load(f)

model_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model.yaml"
)
with open(model_config_path, "r") as f:
    model_config = yaml.safe_load(f)

# Set seed
seed_everything(model_config["seed"])

fold_info = pd.read_parquet(fold_dir / "info.parquet")

# for fold_path in fold_dir.iterdir():

models_dir = fold_dir / "models"
best_model_paths = list(models_dir.glob("**/best_model.ckpt"))

# Make sure the loader keeps track of the concept wise batches
# follow_this = ["y"]
# for i, j in itertools.product(dataset.concept_names, ["x", "edge_index"]):
#     follow_this.append(f"{i}__{j}")

# %%

# Load concept GNN models
model_dict = {}  # Init model dictionary

# Load models
for concept_model_chkpt in best_model_paths:
    concept = concept_model_chkpt.parent.name

    # with open(concept_configs_dir / f'{concept}.yaml', 'r') as f:
    #     concept_config = yaml.safe_load(f)

    # Load dataset
    ds_train = CptDatasetMemo(
        root=fold_dir, fold_info=fold_info, concept=concept, split="train"
    )
    assert ds_train[0].concept == concept

    # Save dataset information to config
    model_config["num_classes"] = ds_train.num_classes
    model_config["in_channels"] = ds_train.num_node_features
    model_config["hidden_channels"] = (
        model_config["in_channels"] * model_config["scaler"]
    )

    # Load model
    model = GNN_plus_MPL(model_config, ds_train)
    module = LitModule.load_from_checkpoint(
        concept_model_chkpt, model=model, config=train_config
    )
    model = module.model

    # note: we could also load just the model
    # model = GNN_plus_MPL(model_config, ds_train)
    # state_dict = torch.load(concept_model_chkpt)['state_dict']
    # state_dict = {key.replace('model.', ''): value for key, value in state_dict.items() if key.startswith('model.')}
    # model.load_state_dict(state_dict)

    # Remove head
    model = model.get_submodule("gnn")

    # Add to dictionary
    model_dict[concept] = model

# check if all models have the same output dimension
n_out = set(model.gnn.out_channels for model in model_dict.values())
assert len(n_out) == 1
n_out = int(n_out.pop())

# Compleat config
# Save embedding size to variable
model_config["emb_size"] = n_out
# model_config["num_classes"] = concept_dataset.num_classes  # this should not have changed
model_config["num_concepts"] = len(best_model_paths)


# Insatiate full model. Concept GNN plus aggregator
graph_concept_learner = GraphConceptLearner(
    concept_learners=nn.ModuleDict(model_dict),
    config=model_config,
    device=device,
)

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


# Save attention maps to file
if gcl_cfg["aggregator"] == "transformer":
    graph_concept_learner.aggregator.return_attention_map()
    generate_and_save_attention_maps(
        model=graph_concept_learner,
        loader=DataLoader(
            dataset=dataset,
            batch_size=1,
            follow_batch=follow_this,
        ),
        device=device,
        follow_this_metrics=follow_this_metrics,
        out_dir=out_dir,
    )

# End run
mlflow.end_run()
