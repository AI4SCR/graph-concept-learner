from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import yaml
from graph_cl.preprocessing.splitV2 import SPLIT_STRATEGIES
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

# paths
experiment_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2"
)
data_dir = Path("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data")

concept_graphs_dir = data_dir / "03_concept_graphs"
folds_dir = experiment_dir / "data" / "05_folds"
folds_dir.mkdir(parents=True, exist_ok=True)

# labels
sample_labels_path = data_dir / "02_processed" / "labels" / "samples"
sample_labels = pd.read_parquet(sample_labels_path)

# configs
attribute_config_path = experiment_dir / "configuration" / "attribute.yaml"
data_config_path = experiment_dir / "configuration" / "data.yaml"

with open(attribute_config_path, "r") as f:
    attribute_config = yaml.load(f, Loader=yaml.FullLoader)

with open(data_config_path, "r") as f:
    data_config = yaml.load(f, Loader=yaml.FullLoader)

# %%
def gather_sample_data() -> pd.DataFrame:
    counts = pd.DataFrame()
    for concept in data_config["concepts"]:
        for graph_path in (concept_graphs_dir / concept).glob("*.pt"):
            core = graph_path.stem
            g = torch.load(graph_path)
            counts.loc[core, concept] = g.num_nodes
    sample_data = counts.join(sample_labels)
    return sample_data


def split_samples_into_folds(valid_samples) -> list[pd.DataFrame]:
    func = SPLIT_STRATEGIES[data_config["split"]["strategy"]]
    folds = func(valid_samples, **data_config["split"]["kwargs"])
    for i, fold in enumerate(folds):
        out_dir = folds_dir / f"fold_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fold.to_parquet(out_dir / f"info.parquet")
    return folds


def sample_has_targets(sample_data) -> pd.DataFrame:
    targets = sample_data[data_config["targets"]]
    # TODO: how to handle nan strings?
    valid_cores = sample_data[(targets.notna()) & (targets != "nan")]
    return valid_cores


def filter_samples(sample_data):
    sample_data = sample_has_targets(sample_data)
    m = (
        sample_data[data_config["concepts"]]
        >= data_config["filter"]["min_cells_per_graph"]
    ).all(1)
    return sample_data[m]


def collect_features():
    feat_config = attribute_config["features"]
    processed_dir = data_dir / "02_processed"
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


def normalize_features(fold_info: pd.DataFrame, output_dir: Path):
    feat = collect_features()

    for split in ["train", "val", "test"]:
        split_feat = feat.loc[fold_info[fold_info.split == split].index, :]
        assert split_feat.index.get_level_values("cell_id").isna().any() == False

        feat_norm = _normalize_features(split, split_feat)
        for core, grp_data in feat_norm.groupby("core"):
            grp_data.to_parquet(output_dir / f"{core}.parquet")


def _normalize_features(split, split_feat):
    # arcsinh transform

    X = split_feat.values.copy()

    cofactor = data_config["normalize"]["cofactor"]
    np.divide(X, cofactor, out=X)
    np.arcsinh(X, out=X)

    # censoring
    censoring = data_config["normalize"]["censoring"]
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
    feat = feat.loc[graph.cell_ids, :]  # align the features with the graph
    graph.x = torch.tensor(feat.values, dtype=torch.float32)
    graph.feature_name = feat.columns.tolist()

    return graph


def attribute_graphs(fold_info: pd.DataFrame, feat_dir: Path, output_dir: Path):
    for concept in data_config["concepts"]:
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


# %%
encoder = LabelEncoder()

# %%

sample_data = gather_sample_data()
valid_samples = filter_samples(sample_data)
encoded_target = encoder.fit_transform(valid_samples[data_config["targets"]])
valid_samples = valid_samples.assign(target=encoded_target)
folds = split_samples_into_folds(valid_samples)

for fold_info_path in folds_dir.glob("*/info.parquet"):
    scaler = MinMaxScaler()
    fold_info = pd.read_parquet(fold_info_path)

    feat_dir = fold_info_path.parent / "normalized_features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    normalize_features(fold_info, feat_dir)

    output_dir = fold_info_path.parent / "attributed_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    attribute_graphs(fold_info, feat_dir, output_dir)

# %%
