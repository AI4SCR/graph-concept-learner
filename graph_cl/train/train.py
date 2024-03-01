from torch_geometric.loader import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from graph_cl.configuration.configurator import TrainConfig


def train(model: L.LightningModule, train_config: TrainConfig, data="asdf"):
    ds_train = data["train"]
    ds_val = data["val"] if data["val"] else None

    dl_train = DataLoader(
        ds_train,
        batch_size=train_config.batch_size,
        shuffle=True,
    )
    dl_val = DataLoader(ds_val, batch_size=train_config.batch_size)

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.tracking.checkpoint_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = L.Trainer(**train_config.trainer.dict(), callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_val)
