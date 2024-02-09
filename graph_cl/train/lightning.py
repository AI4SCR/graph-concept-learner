import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall

from ..configuration.configurator import Training


class LitModule(L.LightningModule):
    def __init__(self, model: nn.Module, config: Training):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.config = config

        self.criterion = self.configure_criterion()
        self.metrics = self.configure_metrics()

    def forward(self, x):
        out = self.model(x)
        y_pred = out.argmax(dim=1)
        return y_pred

    def training_step(self, batch, batch_idx):
        out = self.model(batch)

        # NOTE: instead of forward
        # y_pred = self(batch)
        # y_pred = out.argmax(dim=1)

        loss = self.criterion(out, batch.y)
        # self.metrics(y_pred, batch.y)

        self.log("train_loss", loss)
        # self.log('train_acc', self.metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.criterion(out, batch.y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.criterion(out, batch.y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer_name = self.config.optimizer.name
        optimizer_kwargs = self.config.optimizer.kwargs

        optimizer = getattr(torch.optim, optimizer_name)(
            self.parameters(), **optimizer_kwargs
        )

        scheduler = self.configure_scheduler(optimizer)
        # alternative, use a config
        # scheduler = {
        #     'scheduler': scheduler,
        #     'interval': self.config.scheduler.interval,
        #     'frequency': self.config.scheduler.frequency
        # }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_scheduler(self, optimizer):
        from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

        scheduler_name = self.config.scheduler.name
        scheduler_kwargs = self.config.scheduler.kwargs

        if scheduler_name == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, **scheduler_kwargs)
        else:
            # TODO: LambdaLR
            raise NotImplementedError()

        return scheduler

    def configure_metrics(self):
        metrics_config = {
            "train": {
                "accuracy": Accuracy(task="multiclass", average="macro"),
            },
            "val": {
                "accuracy": Accuracy(task="multiclass"),
                "precision": Precision(task="multiclass"),
                "recall": Recall(task="multiclass"),
            },
        }
        return Accuracy(task="multiclass", average="macro")

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss()
