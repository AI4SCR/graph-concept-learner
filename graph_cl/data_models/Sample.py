import uuid
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from skimage.io import imread
import torch
from pydantic import BaseModel, Field, computed_field
from .MixIns import PickleMixIn
from pathlib import Path


class Sample(BaseModel, PickleMixIn):
    model_config = dict(arbitrary_types_allowed=True)

    # IDs
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str

    # OBSERVATION DATA
    expression_url: None | Path = None
    location_url: None | Path = None
    spatial_url: None | Path = None
    labels_url: None | Path = None

    # SAMPLE DATA
    mask_url: None | Path = None
    img_url: None | Path = None
    metadata_url: None | Path = None
    sample_labels_url: None | Path = None

    # CONCEPT DATA
    concept_graph_url: dict[str, Path] = Field(default_factory=lambda: dict())
    attributes_url: None | Path = None

    # META DATA
    split: None | str = None
    target: None | str = None
    target_encoded: None | int = None

    @property
    def expression(self) -> pd.DataFrame:
        return pd.read_parquet(self.expression_url) if self.expression_url else None

    @property
    def location(self) -> pd.DataFrame:
        return pd.read_parquet(self.location_url) if self.expression_url else None

    @property
    def spatial(self) -> pd.DataFrame:
        return pd.read_parquet(self.spatial_url) if self.spatial_url else None

    @property
    def labels(self) -> pd.DataFrame:
        return pd.read_parquet(self.labels_url) if self.labels_url else None

    @property
    def mask(self) -> np.ndarray:
        return imread(self.mask_url, plugin="tifffile") if self.mask_url else None

    @property
    def img(self) -> np.ndarray:
        return imread(self.img_url, plugin="tifffile") if self.img_url else None

    @property
    def sample_labels(self) -> pd.DataFrame:
        return (
            pd.read_parquet(self.sample_labels_url) if self.sample_labels_url else None
        )

    @property
    def cohort(self) -> str:
        return self.metadata["cohort"]

    def attributed_graph(self, concept_name: str) -> Data:
        assert self.attributes_url is not None
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
