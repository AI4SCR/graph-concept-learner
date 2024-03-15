import uuid
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from skimage.io import imread
import torch
from pydantic import BaseModel, Field
from .MixIns import PickleMixIn, ModelIOMixIn
from pathlib import Path


class Sample(BaseModel, PickleMixIn, ModelIOMixIn):
    model_config = dict(arbitrary_types_allowed=True)

    # IDs
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    # observation_ids: list[tuple] | pd.Index  # TODO: should we add this?

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
            pd.read_parquet(self.sample_labels_url).squeeze()
            if self.sample_labels_url
            else None
        )

    @property
    def metadata(self) -> pd.DataFrame:
        return (
            pd.read_parquet(self.metadata_url).squeeze() if self.metadata_url else None
        )

    @property
    def cohort(self) -> str:
        return self.sample_labels["cohort"]

    @property
    def attributes(self) -> pd.DataFrame:
        return pd.read_parquet(self.attributes_url) if self.attributes_url else None

    @property
    def stage(self) -> str:
        return self.split

    def concept_graph(self, concept_name: str) -> Data:
        return (
            torch.load(self.concept_graph_url[concept_name])
            if concept_name in self.concept_graph_url
            else None
        )

    def attributed_graph(self, concept_name: str) -> Data:
        from ..preprocessing.attribute import attribute_graph

        assert self.attributes_url is not None
        assert self.attributes.isna().any().any() == False

        graph = self.concept_graph(concept_name)
        attrs = self.attributes
        graph = attribute_graph(graph, attrs)

        graph.y = torch.tensor(self.target_encoded, dtype=torch.int32)
        graph.sample_id = self.id
        graph.name = self.name
        graph.cohort = self.cohort
        graph.target = self.target

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

        for concept_name in self.concept_graph_url:
            graph = self.concept_graph(concept_name)
            # check that graph contains subset of object_ids
            assert set([int(i) for i in graph.object_id]) <= object_ids

        if self.attributes:
            assert (self.expression.index == self.attributes.index).all()

        return True
