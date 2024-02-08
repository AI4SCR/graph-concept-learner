import lightning as L
import torch
import torch.nn as nn
from ..configuration.configurator import Training

# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, model: nn.Module, config: Training):
        super().__init__()
        self.model = model
        self.config = config

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.criterion(out, batch.y)
        return loss

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
