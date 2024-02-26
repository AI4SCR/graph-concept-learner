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
from graph_cl.train.lightning import LitGNN, LitGCL

from graph_cl.datasets.ConceptDataset import CptDatasetMemo
from graph_cl.datasets.ConceptSetDataset import ConceptSetDataset
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

model_gcl_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gcl.yaml"
)
with open(model_gcl_config_path, "r") as f:
    model_gcl_config = yaml.safe_load(f)

model_gnn_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gnn.yaml"
)
with open(model_gnn_config_path, "r") as f:
    model_gnn_config = yaml.safe_load(f)

# Set seed
# TODO: this should not be part of the model config. Seed in this config should only be used to seed model init.
seed_everything(model_gcl_config["seed"])

fold_info = pd.read_parquet(fold_dir / "info.parquet")

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
    model_gnn_config["num_classes"] = ds_train.num_classes
    model_gnn_config["in_channels"] = ds_train.num_node_features
    model_gnn_config["hidden_channels"] = (
        model_gnn_config["in_channels"] * model_gnn_config["scaler"]
    )

    # Load model
    model = GNN_plus_MPL(model_gnn_config, ds_train)
    module = LitGNN.load_from_checkpoint(
        concept_model_chkpt, model=model, config=train_config
    )
    model = module.model

    # note: we could also load just the model
    # model = GNN_plus_MPL(model_gnn_config, ds_train)
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

# dataset
fold_info = pd.read_parquet(fold_dir / "info.parquet")
ds_train = ConceptSetDataset(root=fold_dir, fold_info=fold_info, split="train")
ds_val = ConceptSetDataset(root=fold_dir, fold_info=fold_info, split="val")
ds_test = ConceptSetDataset(root=fold_dir, fold_info=fold_info, split="test")

# dataloader
dl_test = DataLoader(ds_test, batch_size=train_config.batch_size)
dl_val = DataLoader(ds_val, batch_size=train_config.batch_size)
dl_train = DataLoader(ds_test, batch_size=train_config.batch_size)

# complete config
# Save embedding size to variable
model_gcl_config["emb_size"] = n_out
model_gcl_config[
    "num_classes"
] = ds_train.num_classes  # note: this should not have changed
model_gcl_config["num_concepts"] = len(best_model_paths)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate full model. Concept GNN plus aggregator
graph_concept_learner = GraphConceptLearner(
    concept_learners=nn.ModuleDict(model_dict),
    config=model_gcl_config,
)

gcl = LitGCL(model=graph_concept_learner, config=train_config)

optim_gnn_kwargs = list(
    filter(
        lambda x: x["layer"] == "concept_learners",
        train_config["optimizer"].get("layers", []),
    )
)
gnn_lr = optim_gnn_kwargs[0]["lr"] if optim_gnn_kwargs else 0

# If the gnns_lr = 0 the freeze parameters in model
if gnn_lr == 0:
    for parameter in gcl.model.concept_learners.parameters():
        parameter.requires_grad = False
    # remove the concept_learners from the optimizer
    train_config["optimizer"]["layers"] = list(
        filter(
            lambda x: x["layer"] != "concept_learners",
            train_config["optimizer"].get("layers", []),
        )
    )

models_dir = fold_dir / "model_gcl"

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=models_dir,
    filename="best_model",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

trainer = L.Trainer(
    limit_train_batches=100, max_epochs=2, callbacks=[checkpoint_callback]
)
trainer.fit(model=gcl, train_dataloaders=dl_train, val_dataloaders=dl_val)
trainer.test(model=gcl, dataloaders=dl_test)

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
