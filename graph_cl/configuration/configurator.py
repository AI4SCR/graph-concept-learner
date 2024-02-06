from pydantic import BaseModel, field_validator
from pathlib import Path

# %%
class Experiment(BaseModel):
    name: str
    description: str
    prediction_target: str
    root: Path

    @field_validator("root", mode="before")
    @classmethod
    def convert_raw_to_path(cls, v):
        return Path(v)


class Data(BaseModel):
    raw: Path
    intermediate: Path

    @field_validator("raw", "intermediate", mode="before")
    @classmethod
    def convert_raw_to_path(cls, v):
        return Path(v)


class Filter(BaseModel):
    min_num_cells_per_graph: int


class Normalize(BaseModel):
    method: str


class Split(BaseModel):
    method: str
    n_folds: int


# NOTE: This could potentially be replaced by pydantic's built-in `BaseSettings` class
class Configuration(BaseModel):
    experiment: Experiment
    data: Data
    filter: Filter
    normalize: Normalize
    split: Split
