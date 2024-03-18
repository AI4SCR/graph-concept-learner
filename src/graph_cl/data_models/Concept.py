from pydantic import BaseModel
from .MixIns import YAMLMixIN


class Graph(BaseModel):
    topology: str
    params: dict


class Filter(BaseModel):
    col_name: str
    include_labels: list[str]


class ConceptConfig(BaseModel, YAMLMixIN):
    name: str
    graph: Graph
    filter: Filter
