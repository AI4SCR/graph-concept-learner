import torch
import torch.nn as nn

import yaml
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.models.graph_concept_learnerV2 import GraphConceptLearner
from graph_cl.train.lightning import LitGCL

from graph_cl.dataloader.ConceptDataModule import ConceptDataModule
from graph_cl.data_models.Model import ModelGNNConfig, ModelGCLConfig
from graph_cl.data_models.Train import TrainConfig

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path
import pandas as pd


# %%


def train_gcl(
    model_gcl_config: ModelGCLConfig,
    concept_models: dict[str, nn.Module],
    datasets: dict[str, ConceptDataModule],
    train_config: TrainConfig,
):
    # Set seed
    # TODO: this should not be part of the model config. Seed in this config should only be used to seed model init.
    seed_everything(model_gcl_config.seed)

    # %% dataloader
    dl_train = DataLoader(datasets["train"], batch_size=train_config.batch_size)
    dl_val = DataLoader(datasets["val"], batch_size=train_config.batch_size)
    dl_test = DataLoader(datasets["test"], batch_size=train_config.batch_size)

    # %% instantiate GCL model. Concept GNN plus aggregator
    graph_concept_learner = GraphConceptLearner(
        concept_learners=nn.ModuleDict(concept_models),
        config=model_gcl_config.dict(),
    )
    gcl = LitGCL(model=graph_concept_learner, config=train_config)

    # %% train
    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.tracking.checkpoint_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = L.Trainer(**train_config.trainer.dict(), callbacks=[checkpoint_callback])
    trainer.fit(model=gcl, train_dataloaders=dl_train, val_dataloaders=dl_val)

    # %% test
    trainer.test(model=gcl, dataloaders=dl_test)


def load_concept_models(
    model_chkpt_paths: list[Path], model_gnn_config: ModelGNNConfig
) -> dict[str, nn.Module]:
    model_dict = {}
    for concept_model_chkpt in model_chkpt_paths:
        # TODO: provide the concept name to the model instantiation instead of inferring it from the path
        concept = concept_model_chkpt.parent.name

        # Load model
        model = GNN_plus_MPL(model_gnn_config.dict())
        state_dict = torch.load(concept_model_chkpt)["state_dict"]
        state_dict = {
            key.replace("model.", ""): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        model.load_state_dict(state_dict)

        # note: we could also load just the model from lit module
        # module = LitGNN.load_from_checkpoint(
        #     concept_model_chkpt, model=model, config=train_config
        # )
        # model = module.model

        # Remove head
        model = model.get_submodule("gnn")

        # Add to dictionary
        model_dict[concept] = model

    # check if all models have the same output dimension
    n_out = set(model.gnn.out_channels for model in model_dict.values())
    assert len(n_out) == 1
    return model_dict


def train_from_files(
    fold_dir: Path,
    train_config_path: Path,
    model_gcl_config_path: Path,
    model_gnn_config_path: Path,
):
    # %% load files from paths
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)
        train_config = TrainConfig(**train_config)

    with open(model_gcl_config_path, "r") as f:
        model_gcl_config = yaml.safe_load(f)
        model_gcl_config = ModelGCLConfig(**model_gcl_config)

    with open(model_gnn_config_path, "r") as f:
        model_gnn_config = yaml.safe_load(f)
        model_gnn_config = ModelGNNConfig(**model_gnn_config)

    gcl_ckpt_dir = fold_dir / "models" / "gcl"
    train_config.tracking.checkpoint_dir = gcl_ckpt_dir

    gnn_ckpt_dir = fold_dir / "models" / "concepts"

    fold_info = pd.read_parquet(fold_dir / "info.parquet")

    # %% Load concept GNN models
    best_model_paths = list(gnn_ckpt_dir.glob("**/best_model.ckpt"))
    model_dict = load_concept_models(best_model_paths, model_gnn_config)

    # %% load dataset
    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = ConceptSetDataset(
            root=fold_dir, fold_info=fold_info, split=split
        )

    train_gcl(model_gcl_config, model_dict, datasets, train_config)
