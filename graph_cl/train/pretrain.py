import sys
import os
import mlflow
import yaml
import torch
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.datasets.ConceptDataset import CptDatasetMemo
from graph_cl.utils.mlflow_utils import robust_mlflow, start_mlflow_run
from graph_cl.utils.train_utils import (
    get_optimizer_class,
    build_scheduler,
    test_and_log_best_models,
    train_validate_and_log_n_epochs,
    get_dict_of_metric_names_and_paths,
)

from pathlib import Path
import pandas as pd


def pretrain_concept(
    root: Path,
    fold_path: Path,
    pretrain_config_path: Path,
    output_dir: Path,
):
    fold = pd.read_csv(fold_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(pretrain_config_path) as file:
        pretrain_config = yaml.load(file, Loader=yaml.Loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    ds_train = CptDatasetMemo(root=root, fold_meta_data=fold, split="train")
    ds_val = CptDatasetMemo(root=root, fold_meta_data=fold, split="val")
    ds_test = CptDatasetMemo(root=root, fold_meta_data=fold, split="test")

    # Save dataset information to config
    pretrain_config["num_classes"] = ds_train.num_classes
    pretrain_config["in_channels"] = ds_train.num_node_features
    pretrain_config["hidden_channels"] = (
        pretrain_config["in_channels"] * pretrain_config["scaler"]
    )

    # Set seed
    seed_everything(pretrain_config["seed"])

    # Build model.
    # Important to pass train_dataset in cpu, not cuda.
    model = GNN_plus_MPL(pretrain_config, ds_train)

    # Move to CUDA if available
    model.to(device)

    dl_train = DataLoader(
        ds_train,
        batch_size=pretrain_config["batch_size"],
        shuffle=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=pretrain_config["batch_size"],
        shuffle=True,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=pretrain_config["batch_size"],
        shuffle=True,
    )

    # Define optimizer
    optimizer_class = get_optimizer_class(pretrain_config)
    optimizer = optimizer_class(model.parameters(), lr=pretrain_config["lr"])

    # Define loss function.
    criterion = torch.nn.CrossEntropyLoss()

    # Define learning rate decay strategy
    scheduler = build_scheduler(pretrain_config, optimizer)

    # Define mlflow experiment
    # if mlflow_on_remote_server == "False":
    #     mlflow.set_tracking_uri(mlflow_uri)
    #
    # start_mlflow_run(root, pred_target, out_dir)
    #
    # pretrain_config["run_type"] = "pretrain_concept"
    # pretrain_config["normalized_with"] = normalized_with
    # pretrain_config["fold"] = os.path.basename(concept_dataset_dir)
    # pretrain_config["split_strategy"] = split_strategy
    # pretrain_config["cfg_id"] = cfg_id
    # pretrain_config["concept"] = os.path.basename(os.path.dirname(concept_dataset_dir))
    # pretrain_config["attribute_config"] = os.path.basename(
    #     os.path.dirname(os.path.dirname(concept_dataset_dir))
    # )
    # pretrain_config["path_input_config"] = pretrain_config_path
    # pretrain_config["path_output_models"] = out_dir
    # pretrain_config["path_input_data"] = os.path.dirname(concept_dataset_dir)
    if pretrain_config["gnn"] == "PNA":
        pretrain_config.pop("deg")

    # Log config
    robust_mlflow(mlflow.log_params, params=pretrain_config)

    # Training and evaluation
    # Log frequency in terms of epochs
    log_every_n_epochs = int(log_frequency)

    # Save checkpoints for the following metrics
    follow_this_metrics = get_dict_of_metric_names_and_paths(out_file_1, out_file_2)

    # Train and validate for cfg["n_epochs"]
    train_validate_and_log_n_epochs(
        cfg=pretrain_config,
        model=model,
        train_loader=dl_train,
        val_loader=dl_val,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        log_every_n_epochs=log_every_n_epochs,
        device=device,
        follow_this_metrics=follow_this_metrics,
    )

    # Load best models an compute test metrics ###
    test_and_log_best_models(
        cfg=pretrain_config,
        model=model,
        test_loader=dl_test,
        criterion=criterion,
        device=device,
        follow_this_metrics=follow_this_metrics,
        out_dir=out_dir,
        split="test",
    )

    # End run
    mlflow.end_run()
