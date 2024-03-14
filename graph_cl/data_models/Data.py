from pydantic import BaseModel
from .MixIns import FromYamlMixIn


class DataConfig(BaseModel, FromYamlMixIn):
    class Filter(BaseModel):
        min_num_nodes: int

    class Split(BaseModel):
        strategy: str
        kwargs: dict

    class Normalize(BaseModel):
        method: str
        kwargs: dict

    dataset_name: str
    target: str
    filter: Filter
    split: Split
    normalize: Normalize
    concepts: list[str]
    features: dict
