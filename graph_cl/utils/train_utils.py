import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
import warnings
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from graph_cl.utils.mlflow_utils import (
    robust_mlflow,
    make_confusion_matrix,
)
import mlflow
import os


# Util functions
def split_concept_dataset(splits_df, index_col, dataset):
    # Read split map
    split_map = pd.read_csv(splits_df, index_col=index_col)

    # Init dictionary of subseted datasets
    subseted_datasets = {}

    # Loop over splits
    for split in split_map["split"].unique():
        # Get sample ids
        ids = set(split_map[split_map["split"] == split].index.values + ".pt")

        # List of indexes of ids in dataset
        idxs = []
        for i, spl in enumerate(dataset.file_names):
            if spl in ids:
                idxs.append(i)

        # Subset dataset
        # NOTE: Subscript operator invokes __getitem__()
        # Here it creates a copy of the dataset and sets the self._indices variable
        # to idxs. Then when [i] is called on the new object the i'th index from
        # the self._indices is returned.
        subseted_datasets[split] = dataset[idxs]

    # return dataset
    return subseted_datasets


def get_optimizer_class(cfg):
    optimizers = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    optimizer_class = optimizers[cfg["optim"]]
    return optimizer_class


def build_scheduler(cfg, optimizer):
    if cfg["scheduler"][0] == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=cfg["scheduler"][1])
    elif cfg["scheduler"][0] == "LambdaLR":
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: cfg["scheduler"][1]
            ** (epoch // cfg["scheduler"][2]),
        )
    else:
        raise Exception(
            "Sorry, only ExponentialLR and LambdaLR are supported as learning rate decay strategies."
        )

    return scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    # Adjust lr
    scheduler.step()


def eval_one_epoch(loader, split: str, model, device, cfg, criterion):
    # Get loss, predictions and true labels
    mean_loss, y_pred, y_true = predict(loader, model, device, criterion)
    (
        balanced_accuracy,
        weighted_precision,
        weighted_recall,
        weighted_f1_score,
    ) = compute_metrics(y_true, y_pred)
    return {
        f"{split}_balanced_accuracy": balanced_accuracy,
        f"{split}_weighted_precision": weighted_precision,
        f"{split}_weighted_recall": weighted_recall,
        f"{split}_weighted_f1_score": weighted_f1_score,
        f"{split}_loss": mean_loss,
    }


def compute_metrics(y_true, y_pred):
    # Set warnings as errors.
    warnings.filterwarnings("error")
    try:
        balanced_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        weighted_precision = precision_score(
            y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0
        )
        # Every split should have both positive and negative classes, hence this two metrics below should never be
        # ill defined (have a zero divisor). Raise warning and show y_pred and y_true if not so.
        weighted_recall = recall_score(
            y_true=y_true, y_pred=y_pred, average="weighted", zero_division="warn"
        )
        weighted_f1_score = f1_score(
            y_true=y_true, y_pred=y_pred, average="weighted", zero_division="warn"
        )
    except Warning:
        # Print labels and predictions for debugging
        print("One of the metrics is ill defined. Run tagged with metric warning.")
        print("Printing y_pred:")
        print(y_pred)
        print("Printing y_true:")
        print(y_true)

        # Log warning
        robust_mlflow(
            mlflow.set_tag, key="metric_warning", value="Run with metric warning"
        )

    # Reset warnings as warnings to avoid stoping the execution.
    warnings.resetwarnings()

    return balanced_accuracy, weighted_precision, weighted_recall, weighted_f1_score


def predict(loader, model, device, criterion):
    # Freeze parameters
    model.eval()

    # Initialize tensor of results
    y_pred = torch.empty(0, dtype=torch.float64, device=device)
    y_true = torch.empty(0, dtype=torch.float64, device=device)
    total_loss = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        # Get predictions
        out = model(data)

        # Use the class with highest probability.
        y_pred = torch.cat((y_pred, out.argmax(dim=1)), 0)

        # Get ground truth labels
        y_true = torch.cat((y_true, data.y), 0)

        loss = criterion(out, data.y)  # Compute the loss.
        assert not torch.isnan(
            loss
        ), f"loss is NAN. Printing y_pred and y_true for batch. y_pred: {out}. y_true: {data.y}. loss: {loss}"
        total_loss += loss.detach().cpu()

    # Compute average loss for the epoch
    mean_loss = total_loss.item() / len(loader)

    # Pass tensors to cpu and int types
    y_pred = y_pred.cpu().int()
    y_true = y_true.cpu().int()

    return mean_loss, y_pred, y_true


def train_validate_and_log_n_epochs(
    cfg,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    log_every_n_epochs,
    device,
    follow_this_metrics,
):
    # for epoch in tqdm(range(1, cfg["n_epoch"] + 1)):  # this line if for debugging
    for epoch in range(1, cfg["n_epoch"] + 1):
        # Train one epoch and obtain loss
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler)

        if epoch % log_every_n_epochs == 0:
            # Evaluate on all splits
            train_metrics = eval_one_epoch(
                train_loader, "train", model, device, cfg, criterion
            )
            val_metrics = eval_one_epoch(
                val_loader, "val", model, device, cfg, criterion
            )

            # Join dictionaries
            metrics = {**train_metrics, **val_metrics}

            # Log performance metris
            robust_mlflow(
                mlflow.log_metrics,
                metrics=metrics,
                step=epoch,
            )
        # Save model to file if metrics in follow_this_metrics are immproved
        for metric, best_so_far_and_path in follow_this_metrics.items():
            # Unpack path and meteric
            best_so_far, out_file = best_so_far_and_path

            if val_metrics[metric] >= best_so_far:
                # Reset best so far
                best_so_far = val_metrics[metric]
                follow_this_metrics[metric][0] = best_so_far

                # Save model
                torch.save(model.state_dict(), out_file)

                # Log performance
                robust_mlflow(
                    mlflow.log_metric,
                    key=f"best_{metric}",
                    value=best_so_far,
                    step=epoch,
                )

                # Log epoch
                robust_mlflow(
                    mlflow.log_metric, key=f"best_{metric}_epoch", value=epoch
                )


def test_and_log_best_models(
    cfg, model, test_loader, criterion, device, follow_this_metrics, out_dir, split
):
    for metric, best_so_far_and_path in follow_this_metrics.items():
        # Unpack values in tuple inside dict
        best_so_far, out_file = best_so_far_and_path

        # Load model and move to device
        model.load_state_dict(torch.load(out_file))
        model.to(device)

        # Compute metrics for the test split
        test_metrics = eval_one_epoch(
            test_loader, f"{split}_best_{metric}", model, device, cfg, criterion
        )

        # Log metrics mlflown
        robust_mlflow(mlflow.log_metrics, metrics=test_metrics)

        # Log confusion matrix
        _, y_pred, y_true = predict(test_loader, model, device, criterion)
        fig = make_confusion_matrix(
            prediction=y_pred.numpy(force=True).astype(int),
            ground_truth=y_true.numpy(force=True).astype(int),
            classes=["positive", "negative"],
        )

        # Define figure name
        name = f"{split}_conf_mat_from_best_{metric}"

        # Save to file
        conf_mat_path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(conf_mat_path)

        # Log on mlflow
        # save_fig_to_mlflow(fig, "confusion_plots", name)


def get_dict_of_metric_names_and_paths(out_file_1, out_file_2):
    follow_this_metrics = {}
    for file in [out_file_1, out_file_2]:
        name_plus_ext = os.path.basename(file)
        name = os.path.splitext(name_plus_ext)[0].split("best_")[1]
        follow_this_metrics[name] = [0, file]
    return follow_this_metrics


def permute_labels(splits_df, pred_target, splitted_datasets):
    # Read split map
    split_map = pd.read_csv(splits_df, index_col="core")

    # Create a new dictionary for the data in each split
    splitted_datasets_in_mem = {}

    # For every split, permute the labels.
    for split in split_map["split"].unique():

        # Randomly permute labels
        split_map.loc[split_map["split"] == split, pred_target] = np.random.permutation(
            split_map.loc[split_map["split"] == split, pred_target]
        )

        # For each sample in the split put it in a list and change its label
        in_mem_data = []
        for i, file_name_ext in enumerate(splitted_datasets[split].file_names):
            file_name = file_name_ext.split(".")[0]
            in_mem_data.append(splitted_datasets[split].get(i))
            in_mem_data[i].y = torch.tensor([split_map.loc[file_name][pred_target]])

        # Save data to dictionary
        splitted_datasets_in_mem[split] = in_mem_data

    # Return permuted data
    return splitted_datasets_in_mem

    # Code for debugging
    # for split in split_map["split"].unique():
    #     for i, file_name_ext in enumerate(splitted_datasets[split].file_names):
    #         print(f"in mem: {splitted_datasets_in_mem[split][i].y.item()}. in disk: {splitted_datasets[split].get(i).y.item()}")
    #         print(f"filename: {file_name_ext}")
    #         assert splitted_datasets[split][i].y.item() == splitted_datasets[split].get(i).y.item()

    # assert False, "All good"
