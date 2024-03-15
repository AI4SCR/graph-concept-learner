from torch import optim, nn
import lightning as L
from lightning import Trainer
import torch
from torch.utils.data import DataLoader


class LitDummy(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)

    def training_step(self, batch, batch_idx):
        return nn.functional.mse_loss(torch.rand(2, 2), torch.rand(2, 2))

    def validation_step(self, batch, batch_idx):
        return nn.functional.mse_loss(torch.rand(2, 2), torch.rand(2, 2))

    def test_step(self, batch, batch_idx):
        return nn.functional.mse_loss(torch.rand(2, 2), torch.rand(2, 2))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Data(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.ds_fit = None
        self.ds_val = None
        self.ds_test = None

    def setup(self, stage):
        if stage == "fit":
            self.ds_fit = [torch.rand(2)] * 10
            self.ds_val = [torch.rand(2)] * 10
        if stage == "validate":
            self.ds_val = [torch.rand(2)] * 10
        if stage == "test":
            self.ds_test = [torch.rand(2)] * 10

    def train_dataloader(self):
        return DataLoader(self.ds_fit)

    def val_dataloader(self):
        return DataLoader(self.ds_val)

    def test_dataloader(self):
        return DataLoader(self.ds_test)


def test_lit_dummy():
    trainer = Trainer()
    trainer.fit(LitDummy(), datamodule=Data())
    trainer.validate(LitDummy(), datamodule=Data())
    trainer.test(LitDummy(), datamodule=Data())
