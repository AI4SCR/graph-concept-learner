from pathlib import Path
import pandas as pd
from torch_geometric.data import Data
import torch


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


def attribute_graph(graph: Data, feat: pd.DataFrame) -> Data:
    assert feat.isna().any().any() == False

    feat.index = feat.index.droplevel("core").astype(int)
    # note: since the concept graph is a subgraph of the full graph, we can assume that the object_ids are a subset of the features
    assert set(graph.object_id).issubset(set(feat.index))

    feat = feat.loc[graph.object_id, :]  # align the features with the graph
    graph.x = torch.tensor(feat.values, dtype=torch.float32)
    graph.feature_name = feat.columns.tolist()

    return graph


def attribute_graphs(
    samples: pd.DataFrame,
    feat: pd.DataFrame,
    concepts: list[str],
    concept_graphs_dir: Path,
):
    for concept in concepts:
        for core in samples.index:
            graph_path = concept_graphs_dir / concept / f"{core}.pt"
            g = torch.load(graph_path)

            g_attr = attribute_graph(g, feat)
            g_attr.y = torch.tensor([samples.loc[core, "target"]])

            g_attr.core = core
            g_attr.concept = concept
