from pathlib import Path
import pandas as pd
from torch_geometric.data import Data
import torch

from ..data_models.Sample import Sample
from ..data_models.Data import FeatureDict


def collect_sample_features(
    sample: Sample, feature_dicts: list[FeatureDict]
) -> pd.DataFrame:
    feats = []
    for feat_dict in feature_dicts:
        attr_name = feat_dict.attribute_name
        include = feat_dict.include
        exclude = feat_dict.exclude
        if include:
            feat = getattr(sample, attr_name)
            if isinstance(include, bool):
                feats.append(feat)
                if exclude:
                    assert set(exclude) < set(feat.columns)
            elif isinstance(include, list):
                feats.append(feat[include])
                if exclude:
                    assert set(exclude) < set(include)  # non empty subset
            if exclude:
                feat = feat.drop(columns=exclude)

    feats = pd.concat(feats, axis=1)
    assert feats.isna().any().any() == False

    return feats


def collect_features(
    samples: list[Sample], feature_dicts: list[FeatureDict]
) -> pd.DataFrame:
    from pandas.api.types import is_numeric_dtype

    feats = []
    for sample in samples:
        feat = collect_sample_features(sample, feature_dicts)
        feats.append(feat)

    feats = pd.concat(feats)
    assert feats.isna().any().any() == False
    assert all([is_numeric_dtype(dtype) for dtype in feats.dtypes])
    return feats


def attribute_graph(graph: Data, feat: pd.DataFrame) -> Data:
    assert feat.isna().any().any() == False

    # note: since the concept graph is a subgraph of the full graph, we can assume that the object_ids are a subset of the features
    assert set([int(i) for i in graph.object_id]).issubset(
        set(feat.index.get_level_values("cell_id"))
    )
    feat = feat.droplevel("core").loc[
        graph.object_id, :
    ]  # align the features with the graph
    graph.x = torch.tensor(feat.values, dtype=torch.float32)
    graph.x_names = feat.columns.tolist()

    return graph
