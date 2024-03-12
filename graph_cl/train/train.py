import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from graph_cl.data_models.Train import TrainConfig


def train(
    model: L.LightningModule, datamodule: L.LightningDataModule, config: TrainConfig
):
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.tracking.checkpoint_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = L.Trainer(**config.trainer.dict(), callbacks=[checkpoint_callback])
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
