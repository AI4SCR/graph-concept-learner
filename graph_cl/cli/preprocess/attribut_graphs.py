from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import yaml
from graph_cl.preprocessing.split import SPLIT_STRATEGIES
from graph_cl.configuration.configurator import DataConfig
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

import click


def gather_sample_counts(concepts: list[str], concept_graphs_dir: Path) -> pd.DataFrame:
    counts = pd.DataFrame()
    for concept in concepts:
        for graph_path in (concept_graphs_dir / concept).glob("*.pt"):
            core = graph_path.stem
            g = torch.load(graph_path)
            counts.loc[core, concept] = g.num_nodes
    return counts


def filter_samples(
    sample_data, concepts: list[str], target: str, min_cells_per_graph: int
) -> pd.DataFrame:
    targets = sample_data[target]
    # TODO: how to handle nan strings?
    valid_cores = sample_data[(targets.notna()) & (targets != "nan")]

    m = (valid_cores[concepts] >= min_cells_per_graph).all(1)

    return valid_cores[m]


def split_samples_into_folds(
    valid_samples, split_strategy: str, **kwargs
) -> list[pd.DataFrame]:
    func = SPLIT_STRATEGIES[split_strategy]
    folds = func(valid_samples, **kwargs)
    return folds


def collect_features(processed_dir: Path, feat_config: dict):
    features = []
    for feat_name, feat_dict in feat_config.items():
        include = feat_dict.get("include", False)
        if include:
            feat_path = processed_dir / "features" / "observations" / feat_name
            feat = pd.read_parquet(feat_path)
            if isinstance(include, bool):
                features.append(feat)
            elif isinstance(include, list):
                features.append(feat[include])

    # add features
    feat = pd.concat(features, axis=1)
    assert feat.isna().any().any() == False

    return feat


def normalize_features(
    feat: pd.DataFrame, fold_info: pd.DataFrame, output_dir: Path, method: str, **kwargs
):
    scaler = MinMaxScaler()
    for split in ["train", "val", "test"]:
        split_feat = feat.loc[fold_info[fold_info.split == split].index, :]
        assert split_feat.index.get_level_values("cell_id").isna().any() == False

        feat_norm = _normalize_features(split, split_feat, scaler=scaler, **kwargs)
        for core, grp_data in feat_norm.groupby("core"):
            grp_data.to_parquet(output_dir / f"{core}.parquet")


def _normalize_features(split, split_feat, scaler, cofactor, censoring):
    X = split_feat.values.copy()

    # arcsinh transform
    np.divide(X, cofactor, out=X)
    np.arcsinh(X, out=X)

    # censoring
    thres = np.quantile(X, censoring, axis=0)
    for idx, t in enumerate(thres):
        X[:, idx] = np.where(X[:, idx] > t, t, X[:, idx])
    if split == "train":
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return pd.DataFrame(X, index=split_feat.index, columns=split_feat.columns)


def _attribute_graph(graph: Data, feat: pd.DataFrame) -> Data:
    assert feat.isna().any().any() == False

    feat.index = feat.index.droplevel("core").astype(int)
    feat = feat.loc[graph.object_id, :]  # align the features with the graph
    graph.x = torch.tensor(feat.values, dtype=torch.float32)
    graph.feature_name = feat.columns.tolist()

    return graph


def _attribute_graphs(
    fold_info: pd.DataFrame,
    feat_dir: Path,
    concepts: list[str],
    concept_graphs_dir: Path,
    output_dir: Path,
):
    for concept in concepts:
        concept_output_dir = output_dir / concept
        concept_output_dir.mkdir(parents=True, exist_ok=True)
        for core in fold_info.index:
            graph_path = concept_graphs_dir / concept / f"{core}.pt"
            g = torch.load(graph_path)

            feat_path = feat_dir / f"{core}.parquet"
            feat = pd.read_parquet(feat_path)
            g_attr = _attribute_graph(g, feat)
            g_attr.y = torch.tensor([fold_info.loc[core, "target"]])

            g_attr.core = core
            g_attr.concept = concept
            torch.save(g_attr, concept_output_dir / f"{core}.pt")


@click.command()
@click.argument("experiment_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
# def attribute_graphs(experiment_dir: Path, concept_graphs_dir: Path, processed_dir: Path):
def attribute_graphs(experiment_dir: Path, data_dir: Path):
    processed_dir = data_dir / "02_processed"
    concept_graphs_dir = data_dir / "03_concept_graphs"
    folds_dir = experiment_dir / "data" / "05_folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    # labels
    sample_labels_path = processed_dir / "labels" / "samples"
    sample_labels = pd.read_parquet(sample_labels_path)

    # configs
    data_config_path = experiment_dir / "configuration" / "data.yaml"

    with open(data_config_path, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
        data_config = DataConfig(**data_config)

    # %%
    encoder = LabelEncoder()

    # %%
    counts = gather_sample_counts(
        concepts=data_config.concepts, concept_graphs_dir=concept_graphs_dir
    )
    sample_data = counts.join(sample_labels)

    valid_samples = filter_samples(
        sample_data,
        target=data_config.target,
        concepts=data_config.concepts,
        min_cells_per_graph=data_config.filter.min_cells_per_graph,
    )

    encoded_target = encoder.fit_transform(valid_samples[data_config.target])
    valid_samples = valid_samples.assign(target=encoded_target)
    folds = split_samples_into_folds(
        valid_samples,
        split_strategy=data_config.split.strategy,
        **data_config.split.kwargs,
    )

    # save fold info
    for i, fold in enumerate(folds):
        out_dir = folds_dir / f"fold_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fold.to_parquet(out_dir / f"info.parquet")

    for fold_info_path in folds_dir.glob("*/info.parquet"):
        fold_info = pd.read_parquet(fold_info_path)

        feat = collect_features(
            processed_dir=processed_dir, feat_config=data_config.features
        )

        feat_norm_dir = fold_info_path.parent / "normalized_features"
        feat_norm_dir.mkdir(parents=True, exist_ok=True)
        normalize_features(
            feat=feat,
            fold_info=fold_info,
            output_dir=feat_norm_dir,
            method=data_config.normalize.method,
            **data_config.normalize.kwargs,
        )

        output_dir = fold_info_path.parent / "attributed_graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        _attribute_graphs(
            fold_info=fold_info,
            feat_dir=feat_norm_dir,
            concepts=data_config.concepts,
            concept_graphs_dir=concept_graphs_dir,
            output_dir=output_dir,
        )
