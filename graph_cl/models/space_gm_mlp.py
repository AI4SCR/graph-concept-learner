### This code is a copy of the code found in this website
### https://gitlab.com/enable-medicine-public/space-gm/-/blob/main/spacegm/models.py
### It is the baseline used in the paper Wu et al. 2022.

import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from graph_cl.utils.train_utils import compute_metrics, build_scheduler
from graph_cl.utils.mlflow_utils import robust_mlflow, make_confusion_matrix
import mlflow


class MLP_pred(object):
    def __init__(
        self,
        n_feat=None,
        n_layers=3,
        scalar=1,
        n_tasks=None,
        gpu=False,
        task="classification",  # classification or cox
        balanced=True,  # used for classification loss
        top_n=0,  # used for cox regression loss
        regularizer_weight=0.005,
        **kwargs,
    ):

        self.n_feat = n_feat
        self.n_layers = n_layers
        self.n_hidden = n_feat * scalar
        self.n_tasks = n_tasks
        self.gpu = gpu
        self.task = task
        self.balanced = balanced
        self.top_n = top_n
        self.regularizer_weight = regularizer_weight

    def build_model(self):
        assert self.n_feat is not None
        assert self.n_tasks is not None
        modules = [torch.nn.Linear(self.n_feat, self.n_hidden)]
        for _ in range(self.n_layers - 1):
            modules.append(torch.nn.LeakyReLU())
            modules.append(torch.nn.Linear(self.n_hidden, self.n_hidden))
        modules.append(torch.nn.LeakyReLU())
        modules.append(torch.nn.Linear(self.n_hidden, self.n_tasks))
        self.model = torch.nn.Sequential(*modules)

        if self.gpu:
            self.model = self.model.cuda()

    def fit(
        self,
        X,
        y,
        X_val,
        y_val,
        follow_this_metrics,
        cfg,
        w=None,
        batch_size=0,
        n_epochs=200,
        lr=0.001,
        log_every_n_epochs=1,
    ):
        assert len(X.shape) == 2
        self.n_feat = X.shape[1]
        if len(y.shape) == 1:
            self.n_tasks = 1
            y = y.reshape((-1, 1))
            w = w.reshape((-1, 1)) if w is not None else w
        else:
            self.n_tasks = y.shape[1]

        batch_size = batch_size if batch_size > 0 else X.shape[0]

        dataset = [torch.from_numpy(X).float(), torch.from_numpy(y).float()]
        if w is not None:
            dataset.append(torch.from_numpy(w).float())
        dataset = TensorDataset(*dataset)
        loader = DataLoader(dataset, batch_size=batch_size)

        self.build_model()
        self.model.zero_grad()
        self.model.train()

        if self.task == "classification":
            pos_weight = (
                (1 - y.mean(0)) / y.mean(0) if self.balanced else np.ones((y.shape[1],))
            )
            self.loss_fn = torch.nn.BCEWithLogitsLoss(
                torch.from_numpy(pos_weight).float()
            )
        elif self.task == "cox":
            self.loss_fn = CoxSGDLossFn(
                top_n=self.top_n, regularizer_weight=self.regularizer_weight
            )
        else:
            raise ValueError("Task type unknown")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Define learning rate decay strategy
        scheduler = build_scheduler(cfg, optimizer)

        for epoch in range(n_epochs):
            for batch in loader:
                if self.gpu:
                    batch = [item.cuda() for item in batch]
                y_pred = self.model(batch[0])
                if self.task == "classification":
                    loss = self.loss_fn(y_pred, batch[1])
                elif self.task == "cox":
                    loss = self.loss_fn(y_pred, batch[1], batch[2])

                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # Adjust lr
            scheduler.step()

            if epoch % log_every_n_epochs == 0:
                train_metrics = self.eval_one_epoch(split="train", X=X, y=y)
                val_metrics = self.eval_one_epoch(split="val", X=X_val, y=y_val)

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

                if val_metrics[metric] > best_so_far:
                    # Reset best so far
                    best_so_far = val_metrics[metric]
                    follow_this_metrics[metric][0] = best_so_far

                    # Save model
                    self.save(out_file)

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

    def predict_proba(self, X, y, batch_size=0):
        assert len(X.shape) == 2
        assert X.shape[1] == self.n_feat

        batch_size = batch_size if batch_size > 0 else X.shape[0]
        dataset = [torch.from_numpy(X).float(), torch.from_numpy(y).float()]
        dataset = TensorDataset(*dataset)
        loader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()

        y_pred = []
        loss = 0
        for batch in loader:
            # print(f"Batch: {batch}")
            if isinstance(batch, list):
                y_true = torch.squeeze(batch[1])
                batch = batch[0]
            if self.gpu:
                batch = batch.cuda()
            # print(f"Batch: {batch.size()}")
            # print(f"y_pred: {y_true.size()}")
            y_pred_batch = torch.squeeze(self.model(batch))
            loss += self.loss_fn(y_pred_batch, y_true)
            y_pred.append(y_pred_batch.cpu().data.numpy())

        mean_loss = loss / len(loader)
        y_pred = np.concatenate(y_pred)
        y_pred = 1 / (1 + np.exp(-y_pred))
        return np.squeeze(np.stack([1 - y_pred, y_pred], -1)), mean_loss

    def predict(self, X, y, **kwargs):
        y_pred, mean_loss = self.predict_proba(X, y, **kwargs)
        return np.argmax(y_pred, -1), mean_loss

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        sd = self.model.state_dict()
        torch.save(sd, path)

    def eval_one_epoch(self, split, X, y):
        y_pred, mean_loss = self.predict(X, y)
        (
            balanced_accuracy,
            weighted_precision,
            weighted_recall,
            weighted_f1_score,
        ) = compute_metrics(y, y_pred)
        metrics = {
            f"{split}_balanced_accuracy": balanced_accuracy,
            f"{split}_weighted_precision": weighted_precision,
            f"{split}_weighted_recall": weighted_recall,
            f"{split}_weighted_f1_score": weighted_f1_score,
            f"{split}_loss": mean_loss,
        }
        return metrics

    def test_and_log_best_models(self, split, X, y, follow_this_metrics, out_dir):
        for metric, best_so_far_and_path in follow_this_metrics.items():
            # Unpack values in tuple inside dict
            best_so_far, out_file = best_so_far_and_path

            # Load model and move to device
            self.load(out_file)
            if self.gpu:
                self.model.cuda()

            metrics = self.eval_one_epoch(split=f"{split}_best_{metric}", X=X, y=y)

            # Log performance metris
            robust_mlflow(mlflow.log_metrics, metrics=metrics)

            # Log confussion matrix
            y_pred, loss = self.predict(X, y)
            fig = make_confusion_matrix(
                prediction=y_pred,
                ground_truth=y,
                classes=["positive", "negative"],
            )

            # Define figure name
            name = f"{split}_conf_mat_from_best_{metric}"

            # Save to file
            conf_mat_path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(conf_mat_path)


class CoxSGDLossFn(torch.nn.Module):
    """Cox SGD loss function"""

    def __init__(self, top_n=2, regularizer_weight=0.05, **kwargs):
        self.top_n = top_n
        self.regularizer_weight = regularizer_weight
        super(CoxSGDLossFn, self).__init__(**kwargs)

    def forward(self, y_pred, length, event):
        assert y_pred.shape[0] == length.shape[0] == event.shape[0]
        n_samples = y_pred.shape[0]
        pair_mat = (
            length.reshape((1, -1)) - length.reshape((-1, 1)) > 0
        ) * event.reshape((-1, 1))

        if self.top_n > 0:
            p_with_rand = pair_mat * (1 + torch.rand_like(pair_mat))
            rand_thr_ind = torch.argsort(p_with_rand, axis=1)[:, -(self.top_n + 1)]
            rand_thr = p_with_rand[(torch.arange(n_samples), rand_thr_ind)].reshape(
                (-1, 1)
            )
            pair_mat = pair_mat * (p_with_rand > rand_thr)

        valid_sample_is = torch.nonzero(pair_mat.sum(1)).flatten()
        pair_mat[(valid_sample_is, valid_sample_is)] = 1

        score_diff = y_pred.reshape((1, -1)) - y_pred.reshape((-1, 1))
        score_diff_row_max = torch.max(score_diff, axis=1, keepdims=True).values
        loss = (torch.exp(score_diff - score_diff_row_max) * pair_mat).sum(1)
        loss = (
            score_diff_row_max[:, 0][valid_sample_is] + torch.log(loss[valid_sample_is])
        ).sum()

        regularizer = torch.abs(pair_mat.sum(0) * y_pred.flatten()).sum()
        loss += self.regularizer_weight * regularizer
        return loss
