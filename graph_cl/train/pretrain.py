import yaml
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.train.lightning import LitModule

from graph_cl.datasets.ConceptDataset import CptDatasetMemo
from graph_cl.configuration.configurator import Training

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path
import pandas as pd


def pretrain_concept(
    root: Path, fold_meta_data: pd.DataFrame, model_config: dict, train_config: Training
):
    # Load dataset
    ds_train = CptDatasetMemo(root=root, fold_meta_data=fold_meta_data, split="train")
    ds_val = CptDatasetMemo(root=root, fold_meta_data=fold_meta_data, split="val")
    ds_test = CptDatasetMemo(root=root, fold_meta_data=fold_meta_data, split="test")

    # Save dataset information to config
    model_config["num_classes"] = ds_train.num_classes
    model_config["in_channels"] = ds_train.num_node_features
    model_config["hidden_channels"] = (
        model_config["in_channels"] * model_config["scaler"]
    )

    # Set seed
    seed_everything(model_config["seed"])

    # Build model.
    # Important to pass train_dataset in cpu, not cuda.
    model = GNN_plus_MPL(model_config, ds_train)
    module = LitModule(model, train_config)

    dl_train = DataLoader(
        ds_train,
        batch_size=train_config.batch_size,
        shuffle=True,
    )
    dl_val = DataLoader(ds_val, batch_size=train_config.batch_size)
    dl_test = DataLoader(ds_test, batch_size=train_config.batch_size)

    if model_config["gnn"] == "PNA":
        train_config.pop("deg")

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.tracking.checkpoint_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = L.Trainer(
        limit_train_batches=100, max_epochs=2, callbacks=[checkpoint_callback]
    )
    trainer.fit(model=module, train_dataloaders=dl_train, val_dataloaders=dl_val)
    trainer.test(model=module, dataloaders=dl_test)


def pretrain_concept_from_files(
    root: Path,
    fold_meta_data_path: Path,
    model_config_path: Path,
    pretrain_config_path: Path,
):
    fold_meta_data = pd.read_csv(fold_meta_data_path)

    with open(pretrain_config_path) as file:
        pretrain_config = yaml.load(file, Loader=yaml.Loader)
        pretrain_config = Training(**pretrain_config)

    with open(model_config_path) as file:
        model_config = yaml.load(file, Loader=yaml.Loader)

    pretrain_concept(
        root,
        fold_meta_data=fold_meta_data,
        model_config=model_config,
        train_config=pretrain_config,
    )
