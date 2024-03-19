import pandas as pd
import torch
from torch_geometric.data import Data

from .normalize import Normalizer
from ..data_models.Data import FeatureDict, DataConfig
from ..data_models.Experiment import GCLExperiment
from ..data_models.Sample import Sample


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


def prepare_attributes(
    splits: dict[str, list[Sample]], experiment: GCLExperiment, config: DataConfig
):
    normalize = Normalizer(**config.normalize.kwargs)

    feats = collect_features(samples=splits["fit"], feature_dicts=config.features)

    # fit the normalizer once on the fit data
    normalize.fit(feats)

    for stage in ["fit", "val", "test", "predict"]:
        if stage not in splits:
            continue

        (experiment.attributes_dir / stage).mkdir(parents=True, exist_ok=True)

        samples = splits[stage]
        for s in samples:
            # enforce that samples are labelled with the correct stage they belong to
            assert s.stage == stage
            if s.attributes is None:
                # collect all features of the given samples
                sample_feats = collect_sample_features(s, feature_dicts=config.features)
                # normalize the features
                attributes = normalize.transform(sample_feats)

                # save the attributes
                attributes_url = experiment.get_attribute_path(stage, s.name)
                s.attributes_url = attributes_url
                attributes.to_parquet(s.attributes_url)

                # update the sample
                sample_path = experiment.get_sample_path(
                    stage=stage, sample_name=s.name
                )
                s.model_dump_to_json(sample_path)
