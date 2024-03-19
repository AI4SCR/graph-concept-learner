from pydantic import BaseModel, field_validator
from .MixIns import YAMLMixIN


class Filter(BaseModel):
    min_num_nodes: int


class Split(BaseModel):
    strategy: str
    kwargs: dict


class Normalize(BaseModel):
    method: str
    kwargs: dict


class FeatureDict(BaseModel):
    """Feature dictionary for feature configuration.

    Args:
        name: name of the feature configuration
        attribute_name: name of the attribute in the `Sample` object
        include: either `bool` if all columns should be included or a list of column names to include.
        exclude: list of column names to exclude, requires include to be not `False`
    """

    name: str
    attribute_name: str
    include: bool | list[str]
    exclude: list[str] | None = None

    # TODO: add validation that include is not False if exclude is not None
    # @field_validator("exclude")
    # def include_not_false(self):


class DataConfig(BaseModel, YAMLMixIN):
    dataset_name: str
    concepts: list[str]
    target: str
    filter: Filter
    split: Split
    normalize: Normalize
    features: list[FeatureDict]
    seed: int = 42
