from pathlib import Path

from pydantic import BaseModel

from .MixIns import FromYamlMixIn


class Tracking(BaseModel):
    mlflow_uri: str | None = None
    checkpoint_dir: str | Path = None


class Optimizer(BaseModel):
    class Layer(BaseModel):
        name: str
        freeze: bool
        kwargs: dict

    name: str
    kwargs: dict
    layers: list[Layer] = []


class Scheduler(BaseModel):
    name: str
    kwargs: dict
    interval: int | str = 1
    frequency: int = 1


class Dataloader(BaseModel):
    batch_size: int = 2
    num_workers: int = 1


class Trainer(BaseModel):
    max_epochs: int
    limit_train_batches: int | float = 1.0
    fast_dev_run: bool = False


class TrainConfig(BaseModel, FromYamlMixIn):
    seed: int = 42
    tracking: Tracking
    optimizer: Optimizer
    scheduler: Scheduler
    dataloader: Dataloader
    trainer: Trainer
