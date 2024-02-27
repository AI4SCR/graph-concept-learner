import logging

import yaml
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.train.lightning import LitGNN

from graph_cl.datasets.ConceptDataset import CptDatasetMemo
from graph_cl.configuration.configurator import TrainConfig, ModelGNNConfig

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path
import pandas as pd


def pretrain_concept(
    concept: str,
    root: Path,
    fold_info: pd.DataFrame,
    model_config: ModelGNNConfig,
    train_config: TrainConfig,
):

    model_root = root / "models" / "concepts" / concept
    logging.info(f"Pretraining model root {model_root}")

    # Load dataset
    ds_train = CptDatasetMemo(
        root=root, fold_info=fold_info, concept=concept, split="train"
    )
    assert ds_train[0].concept == concept

    ds_val = CptDatasetMemo(
        root=root, fold_info=fold_info, concept=concept, split="val"
    )
    assert ds_val[0].concept == concept

    ds_test = CptDatasetMemo(
        root=root, fold_info=fold_info, concept=concept, split="test"
    )
    assert ds_test[0].concept == concept

    # Save dataset information to config
    # TODO: refactor model codebase to work with config objects and not dicts
    model_config = model_config.dict()
    model_config["num_classes"] = ds_train.num_classes
    model_config["in_channels"] = ds_train.num_node_features
    model_config["hidden_channels"] = (
        model_config["in_channels"] * model_config["scaler"]
    )

    # Set seed
    # TODO: do we need a model seed and a train seed?
    #   if so, we need to use the model seed in the model instantiation
    seed_everything(train_config.seed)

    model = GNN_plus_MPL(model_config, ds_train)
    module = LitGNN(model, train_config)

    dl_train = DataLoader(
        ds_train,
        batch_size=train_config.batch_size,
        shuffle=True,
    )
    dl_val = DataLoader(ds_val, batch_size=train_config.batch_size)
    dl_test = DataLoader(ds_test, batch_size=train_config.batch_size)

    if model_config["gnn"] == "PNA":
        raise NotImplementedError()
        # TODO: this does not work anymore
        train_config.pop("deg")

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_root,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = L.Trainer(**train_config.trainer.dict(), callbacks=[checkpoint_callback])
    trainer.fit(model=module, train_dataloaders=dl_train, val_dataloaders=dl_val)
    trainer.test(model=module, dataloaders=dl_test)


def pretrain_concept_from_files(
    concept: str,
    fold_path: Path,
    model_config_path: Path,
    train_config_path: Path,
):
    fold_info = pd.read_parquet(fold_path / "info.parquet")
    assert concept in [i.name for i in (fold_path / "attributed_graphs").glob("*")]

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
        model_config = ModelGNNConfig(**model_config)

    with open(train_config_path, "r") as f:
        pretrain_config = yaml.safe_load(f)
        pretrain_config = TrainConfig(**pretrain_config)

    pretrain_concept(
        concept=concept,
        root=fold_path,
        fold_info=fold_info,
        model_config=model_config,
        train_config=pretrain_config,
    )
