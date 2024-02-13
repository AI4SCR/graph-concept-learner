import pandas as pd
from pathlib import Path
import torch
from torch_geometric.data import Data


def attribute_graph(
    core: str, graph: Data, attribute_config: dict, processed_dir: Path
) -> Data:
    feat_config = attribute_config["features"]
    features = []
    for feat_name, feat_dict in feat_config.items():
        include = feat_dict.get("include", False)
        if include:
            with open(
                processed_dir
                / "features"
                / "observations"
                / feat_name
                / f"{core}.parquet",
                "rb",
            ) as f:
                feat = pd.read_parquet(f)

            if isinstance(include, bool):
                features.append(feat)
            elif isinstance(include, list):
                features.append(feat[include])

    # add features
    feat = pd.concat(features, axis=1)
    assert feat.isna().any().any() == False

    feat = feat.loc[graph.cell_ids, :]  # align the features with the graph
    graph.x = torch.tensor(feat.values, dtype=torch.float32)
    graph.feature_name = feat.columns.tolist()

    # add labels
    label_names = attribute_config["labels"]
    with open(processed_dir / "labels" / "samples" / f"{core}.parquet", "rb") as f:
        labels = pd.read_parquet(f).squeeze()[label_names].tolist()

    graph.labels = labels
    graph.label_names = label_names
    return graph


def attribute_graph_from_files(
    graph_path: Path,
    data_dir: Path,
    attribute_config_path: Path,
    output_dir: Path,
):
    import yaml

    core = graph_path.stem

    with open(attribute_config_path) as f:
        attribute_config = yaml.load(f, Loader=yaml.Loader)

    graph = torch.load(graph_path)
    graph_attributed = attribute_graph(
        core=core,
        graph=graph,
        attribute_config=attribute_config,
        processed_dir=data_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{core}.pt"

    torch.save(graph_attributed, output_path)
