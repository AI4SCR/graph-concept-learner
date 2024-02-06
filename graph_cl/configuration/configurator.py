from pydantic import BaseModel, field_validator
from pathlib import Path

# %%


class Data(BaseModel):
    raw: Path
    intermediate: Path

    @field_validator("raw", "intermediate", mode="before")
    @classmethod
    def convert_raw_to_path(cls, v):
        return Path(v)


# NOTE: This could potentially be replaced by pydantic's built-in `BaseSettings` class
class Configuration(BaseModel):
    data: Data
