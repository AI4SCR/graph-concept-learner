from pydantic import BaseModel, field_validator
from pathlib import Path


# %%
class Project(BaseModel):
    name: str
    description: str
    root: Path

    @field_validator("root", mode="before")
    @classmethod
    def convert_raw_to_path(cls, v):
        return Path(v)


class Experiment(BaseModel):
    name: str
    description: str
    target: str


class Filter(BaseModel):
    min_cells_per_graph: int


class Normalize(BaseModel):
    method: str
    cofactor: int
    censoring: float


class Split(BaseModel):
    method: str
    n_folds: int
    train_size: float
    test_size: float
    val_size: float


class Processing(BaseModel):
    filter: Filter
    normalize: Normalize
    split: Split


class Data(BaseModel):
    raw_dir: Path
    processed_dir: Path
    processing: Processing

    @field_validator("root", mode="before")
    @classmethod
    def convert_raw_to_path(cls, v):
        return Path(v)


# NOTE: This could potentially be replaced by pydantic's built-in `BaseSettings` class
class Configuration(BaseModel):
    project: Project
    experiment: Experiment
    data: Data


class Tracking(BaseModel):
    mlflow_uri: str | None = None
    checkpoint_dir: str


class Optimizer(BaseModel):
    name: str
    kwargs: dict


class Scheduler(BaseModel):
    name: str
    kwargs: dict
    interval: int | str = 1
    frequency: int = 1


class Training(BaseModel):
    tracking: Tracking
    optimizer: Optimizer
    scheduler: Scheduler
    batch_size: int


class Model(BaseModel):
    GNN: None
    MLP: None
