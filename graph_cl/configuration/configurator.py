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
    strategy: str
    kwargs: dict


class Processing(BaseModel):
    filter: Filter
    normalize: Normalize
    split: Split


class Data(BaseModel):
    raw_dir: Path
    processed_dir: Path
    processing: Processing

    @field_validator("raw_dir", "processed_dir", mode="before")
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


class Trainer(BaseModel):
    max_epochs: int
    limit_train_batches: int | float = 1.0
    fast_dev_run: bool = False


class Dataloader(BaseModel):
    batch_size: int = 2
    num_workers: int = 1


class TrainConfig(BaseModel):
    seed: int = 1
    tracking: Tracking
    optimizer: Optimizer
    scheduler: Scheduler
    dataloader: Dataloader
    trainer: Trainer


class Model(BaseModel):
    GNN: None
    MLP: None


class Graph(BaseModel):
    topology: str
    params: dict


class ConceptConfig(BaseModel):
    class Filter(BaseModel):
        col_name: str
        include_labels: list[str]

    name: str
    graph: Graph
    filter: Filter


class DataConfig(BaseModel):
    class Filter(BaseModel):
        min_num_nodes: int

    class Split(BaseModel):
        strategy: str
        kwargs: dict

    class Normalize(BaseModel):
        method: str
        kwargs: dict

    target: str
    filter: Filter
    split: Split
    normalize: Normalize
    concepts: list[str]
    features: dict


from typing import Literal


class ModelGNNConfig(BaseModel):
    act: str
    act_first: bool
    dropout: bool
    gnn: str = "GIN"
    norm: str = "BatchNorm"

    num_classes: int  # task dependent
    num_layers: int = 2
    scaler: int = 2
    in_channels: int  # data dependent
    num_layers_MLP: int = 2

    pool: str = "global_add_pool"
    seed: int = 2


class ModelGCLConfig(ModelGNNConfig):
    aggregator: Literal["transformer", "linear", "concat"] = "transformer"
    mlp_num_layers: int = 2
    mlp_act_key: Literal["relu", "tanh"] = "relu"
    n_heads: int = 8
    depth: int = 1
