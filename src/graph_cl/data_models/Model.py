from typing import Literal

from pydantic import BaseModel
from .MixIns import FromYamlMixIn


class ModelGNNConfig(BaseModel, FromYamlMixIn):
    name: str = "gnn"
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


# TODO: GCL should not be a superset of GNN, why would we need to define the gnn attribute for example?
class ModelGCLConfig(ModelGNNConfig, FromYamlMixIn):
    name: str = "gcl"
    concepts: list[str]
    aggregator: Literal["transformer", "linear", "concat"] = "transformer"
    mlp_num_layers: int = 2
    mlp_act_key: Literal["relu", "tanh"] = "relu"
    n_heads: int = 8
    depth: int = 1
