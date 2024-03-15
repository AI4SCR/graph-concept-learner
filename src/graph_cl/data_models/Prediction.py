from pydantic import BaseModel

from graph_cl.data_models.Sample import Sample


class Prediction(BaseModel):
    sample: Sample
    prediction: str
