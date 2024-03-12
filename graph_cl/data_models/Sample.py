import uuid
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from skimage.io import imread
import torch
from pydantic import BaseModel, Field
from .MixIns import PickleMixIn


class Sample(BaseModel, PickleMixIn):
    model_config = dict(arbitrary_types_allowed=True)

    # TODO: should we load the data from the files, i.e. use expression_url?
    #   one could also use two fields, expression and expression_url and load the data in the computed_field if
    #   necessary
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str

    expression: pd.DataFrame  # this could also be a list of Observations, or we introduce a field that converts to it
    location: pd.DataFrame
    spatial: pd.DataFrame
    labels: pd.DataFrame

    img_url: None | str = None
    mask_url: str
    metadata: pd.Series

    concept_graphs: dict[str, Data] = Field(default_factory=lambda: dict())

    split: None | str = None
    target: None | str = None
    target_encoded: None | int = None

    attributes: None | pd.DataFrame = None

    def attributed_graph(self, concept_name: str) -> Data:
        assert self.attributes is not None
        assert self.attributes.isna().any().any() == False

        graph = self.concept_graphs[concept_name]
        attrs = self.attributes
        # note: since the concept graph is a subgraph of the full graph, we can assume that the object_ids are a subset of the features
        assert set([int(i) for i in graph.object_id]).issubset(
            set(attrs.index.get_level_values("cell_id"))
        )

        attrs = attrs.droplevel("core").loc[
            graph.object_id, :
        ]  # align the features with the graph
        graph.x = torch.tensor(attrs.values, dtype=torch.float32)
        graph.y = torch.tensor(self.target_encoded, dtype=torch.int32)
        graph.feature_name = attrs.columns.tolist()

        return graph

    def check_integrity(self) -> bool:
        # note: we could also compare sets, but we want to enforce the same order of observations
        assert (
            len(self.expression)
            == len(self.location)
            == len(self.spatial)
            == len(self.labels)
        )
        assert (self.expression.index == self.location.index).all()
        assert (self.expression.index == self.spatial.index).all()
        assert (self.expression.index == self.labels.index).all()

        object_ids = set(self.mask.flatten())
        object_ids.remove(0)
        assert set(self.expression.index.get_level_values("cell_id")) == object_ids

        if self.concept_graphs:
            for concept, graph in self.concept_graphs.items():
                # check that graph contains subset of object_ids
                assert set([int(i) for i in graph.object_id]) <= object_ids

        if self.attributes:
            assert (self.expression.index == self.attributes.index).all()

        return True

    @property
    def cohort(self) -> str:
        return self.metadata["cohort"]

    @property
    def img(self) -> np.ndarray:
        return imread(self.img_url, plugin="tifffile")

    @property
    def mask(self) -> np.ndarray:
        return imread(self.mask_url, plugin="tifffile")
