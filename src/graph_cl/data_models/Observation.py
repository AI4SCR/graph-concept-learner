import pandas as pd
from pydantic import BaseModel

from graph_cl.data_models.Sample import Sample


class Observation(BaseModel):
    model_config = dict(arbitrary_types_allowed=True)

    name: str
    sample: Sample
    expression: pd.DataFrame
    location: pd.DataFrame
    spatial: pd.DataFrame
