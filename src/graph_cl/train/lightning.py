from pathlib import Path

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from ..data_models.Train import TrainConfig
from ..models.gnn import GNN_plus_MPL


class LitBase(L.LightningModule):
    def __init__(self, model: nn.Module, config: TrainConfig):
        super().__init__()
        # self.save_hyperparameters()

        self.model = model
        self.config = config

        self.criterion = self.configure_criterion()
        self.metrics = self.configure_metrics()

    def configure_optimizers(self):
        config_optim = self.config.optimizer
        # for key, val in config_optim.items():

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
        from torch.optim.lr_scheduler import ExponentialLR

        scheduler_name = self.config.scheduler.name
        scheduler_kwargs = self.config.scheduler.kwargs

        if scheduler_name == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, **scheduler_kwargs)
        else:
            # TODO: LambdaLR
            raise NotImplementedError()

        return scheduler

    def configure_metrics(self):
        # metrics_config = {
        #     "train": {
        #         "accuracy": Accuracy(task="multiclass", average="macro"),
        #     },
        #     "val": {
        #         "accuracy": Accuracy(task="multiclass"),
        #         "precision": Precision(task="multiclass"),
        #         "recall": Recall(task="multiclass"),
        #     },
        # }
        return Accuracy(task="multiclass", average="macro", num_classes=2)

    def configure_criterion(self):
        return nn.CrossEntropyLoss()


class LitGNN(LitBase):
    name = "LitGNN"

    def __init__(self, model_config: dict, train_config: TrainConfig):
        model = GNN_plus_MPL(model_config)
        super().__init__(model=model, config=train_config)
        self.save_hyperparameters()

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

        self.save_prediction(out, batch, batch_idx)

    def save_prediction(self, predictions, batch, batch_idx):
        yhat = predictions.argmax(dim=1)
        predictions = yhat.detach().cpu().numpy()

        df = pd.DataFrame(
            {
                "prediction": predictions,
                "target": batch.y.cpu().numpy(),
                "sample_id": batch.sample_id,
                "sample_name": batch.sample_name,
                "batch_idx": batch_idx,
            }
        )

        for grp_name, grp in df.groupby("sample_name"):
            grp.to_parquet(self.config.tracking.predictions_dir / f"{grp_name}.parquet")


def load_concept_models(concept_graph_ckpts) -> dict[str, nn.Module]:
    model_dict = {}
    for concept_name, path_chkpt in concept_graph_ckpts.items():
        # # Load model
        # model = GNN_plus_MPL(model_gnn_config.dict())
        # state_dict = torch.load(concept_model_chkpt)["state_dict"]
        # state_dict = {
        #     key.replace("model.", ""): value
        #     for key, value in state_dict.items()
        #     if key.startswith("model.")
        # }
        # model.load_state_dict(state_dict)

        # note: we could also load just the model from lit module
        module = LitGNN.load_from_checkpoint(path_chkpt)
        model = module.model

        # Remove head
        model = model.get_submodule("gnn")

        model_dict[concept_name] = model

    # check if all models have the same output dimension
    n_out = set(model.gnn.out_channels for model in model_dict.values())
    assert len(n_out) == 1
    return model_dict


class LitGCL(LitBase):
    name = "LitGCL"

    def __init__(
        self,
        concept_graph_ckpts: dict[str, Path],
        model_config: dict,
        train_config: TrainConfig,
    ):
        from ..models.graph_concept_learnerV2 import GraphConceptLearner

        concept_models = load_concept_models(concept_graph_ckpts=concept_graph_ckpts)
        model = GraphConceptLearner(
            concept_learners=nn.ModuleDict(concept_models),
            config=model_config,
        )

        super().__init__(model=model, config=train_config)
        self.save_hyperparameters()

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        concept = self.model.concept_names[0]
        y = batch[concept].y

        out = self.model(batch)

        # NOTE: instead of forward
        # y_pred = self(batch)
        # y_pred = out.argmax(dim=1)

        loss = self.criterion(out, y)
        # self.metrics(y_pred, batch.y)

        self.log("train_loss", loss)
        # self.log('train_acc', self.metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        concept = self.model.concept_names[0]
        y = batch[concept].y

        out = self.model(batch)

        loss = self.criterion(out, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        concept = self.model.concept_names[0]
        y = batch[concept].y

        out = self.model(batch)

        loss = self.criterion(out, y)
        self.log("test_loss", loss)

        self.save_prediction(out, batch, batch_idx)

    def save_prediction(self, predictions, batch, batch_idx):
        concept = self.model.concept_names[0]
        concept_batch = batch[concept]
        yhat = predictions.argmax(dim=1)
        predictions = yhat.detach().cpu().numpy()

        df = pd.DataFrame(
            {
                "prediction": predictions,
                "target": concept_batch.y.cpu().numpy(),
                "sample_id": concept_batch.sample_id,
                "sample_name": concept_batch.sample_name,
                "batch_idx": batch_idx,
            }
        )

        for grp_name, grp in df.groupby("sample_name"):
            grp.to_parquet(self.config.tracking.predictions_dir / f"{grp_name}.parquet")

    def configure_optimizers(self):
        optimizer_layers = self.config.optimizer.layers

        if optimizer_layers:
            config_optim = self.config.optimizer
            optimizer_name = config_optim.name
            optimizer_kwargs = config_optim.kwargs  # default kwargs

            optims = []
            for layer in config_optim.layers:
                layer_name = layer.name

                if layer.freeze:
                    for name, param in self.model.named_parameters():
                        # note: this should work even for nested layers where name is like "layer1.layer2.layer3"
                        #   it would be enough to specify "layer2" to freeze all parameters in layer2 even though the full name is "layer1.layer2"
                        #   the questions is: is this robust enough?
                        #   an alternative could be to use getattr(self.model, layer_name).parameters() which would require to specify the full layer name
                        if layer_name in name:
                            param.requires_grad = False
                else:
                    layer_model = getattr(self.model, layer_name)
                    optims.append({"params": layer_model.parameters(), **layer.kwargs})

            optimizer = getattr(torch.optim, optimizer_name)(optims, **optimizer_kwargs)
            scheduler = self.configure_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            super().configure_optimizers()
