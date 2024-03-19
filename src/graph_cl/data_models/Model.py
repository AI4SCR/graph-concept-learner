from typing import Literal

from ai4bmr_core.data_models import MixIns
from pydantic import BaseModel


class GINModel(BaseModel):
    in_channels: None | int = None  # data dependent
    hidden_channels: None | int  # determined by in_channels and scaler
    num_layers: int
    out_channels: None | int = None  # determined by in_channels and scaler
    dropout: bool
    act: str
    act_first: bool
    act_kwargs: None | dict
    norm: str
    norm_kwargs: None | dict
    jk: None


class GNNModel(BaseModel):
    name: str
    kwargs: GINModel
    pool: str
    scaler: int


class MLPModel(BaseModel):
    in_channels: None | int  # determined by hidden_channels of GNN
    num_layers: int
    out_channels: None | int  # determined by num_classes of dataset


class ModelGNNConfig(BaseModel, MixIns.YamlIO):
    name: str = "gnn"
    GNN: GNNModel
    MLP: MLPModel
    seed: int = 2


# TODO: GCL should not be a superset of GNN, why would we need to define the gnn attribute for example?
class ModelGCLConfig(BaseModel, MixIns.YamlIO):
    name: str = "gcl"
    aggregator: Literal["transformer", "linear", "concat"] = "transformer"
    mlp_num_layers: int = 2
    mlp_act_key: Literal["relu", "tanh"] = "relu"
    n_heads: int = 8
    depth: int = 1

    act: str = "ReLU"
    act_first: bool = False
    dropout: bool = False
    gnn: str = "GIN"
    jk: None = None
    norm: str = "BatchNorm"
    num_layers: int = 2
    num_classes: None | int = None
    in_channels: None | int = None
    scaler: int = 4
    num_layers_MLP: int = 2
    pool: str = "global_add_pool"
    seed: int = 2
