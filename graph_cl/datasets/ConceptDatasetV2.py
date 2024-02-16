from typing import List
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import os.path as osp
from torch_geometric.data import Dataset
import yaml
from graph_cl.graph_builder.attribute_graph import attribute_graph
from graph_cl.preprocessing.splitV2 import SPLIT_STRATEGIES
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


class ConceptDataset(Dataset):
    def __init__(
        self,
        experiment_dir: Path,
        data_dir: Path,
    ) -> None:

        # paths
        self.experiment_dir = experiment_dir
        self.data_dir = data_dir

        self.concept_graphs_dir = self.data_dir / "03_concept_graphs"
        self.folds_dir = self.experiment_dir / "data" / "05_folds"

        # labels
        self.sample_labels_path = self.data_dir / "02_processed" / "labels" / "samples"
        self.sample_labels = pd.read_parquet(self.sample_labels_path)

        # configs
        self.attribute_config_path = (
            self.experiment_dir / "configuration" / "attribute.yaml"
        )
        self.data_config_path = self.experiment_dir / "configuration" / "data.yaml"

        with open(self.attribute_config_path, "r") as f:
            self.attribute_config = yaml.load(f, Loader=yaml.FullLoader)

        with open(self.data_config_path, "r") as f:
            self.data_config = yaml.load(f, Loader=yaml.FullLoader)

        self.process()

        super().__init__()

    def gather_sample_data(self) -> pd.DataFrame:
        counts = pd.DataFrame()
        for concept in self.data_config["concepts"]:
            for graph_path in (self.concept_graphs_dir / concept).glob("*.pt"):
                core = graph_path.stem
                g = torch.load(graph_path)
                counts.loc[core, concept] = g.num_nodes
        sample_data = counts.join(self.sample_labels)
        return sample_data

    def sample_has_targets(self, sample_data) -> pd.DataFrame:
        targets = sample_data[self.data_config["targets"]]
        # TODO: how to handle nan strings?
        valid_cores = sample_data[(targets.notna()) & (targets != "nan")]
        return valid_cores

    def filter_samples(self, sample_data):
        sample_data = self.sample_has_targets(sample_data)
        m = (
            sample_data[self.data_config["concepts"]]
            >= self.data_config["filter"]["min_cells_per_graph"]
        ).all(1)
        return sample_data[m]

    def collect_features(self):
        feat_config = self.attribute_config["features"]
        processed_dir = self.data_dir / "02_processed"
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

    def normalize_features(self, fold_info: pd.DataFrame, output_dir: Path):
        self.scaler = MinMaxScaler()
        feat = self.collect_features()

        for split in ["train", "val", "test"]:
            split_feat = feat.loc[fold_info[fold_info.split == split].index, :]
            assert split_feat.index.get_level_values("cell_id").isna().any() == False

            feat_norm = self._normalize_features(split, split_feat)
            for core, grp_data in feat_norm.groupby("core"):
                grp_data.to_parquet(output_dir / f"{core}.parquet")

    def _normalize_features(self, split, split_feat):
        # arcsinh transform

        X = split_feat.values.copy()

        cofactor = self.data_config["normalize"]["cofactor"]
        np.divide(X, cofactor, out=X)
        np.arcsinh(X, out=X)

        # censoring
        censoring = self.data_config["normalize"]["censoring"]
        thres = np.quantile(X, censoring, axis=0)
        for idx, t in enumerate(thres):
            X[:, idx] = np.where(X[:, idx] > t, t, X[:, idx])
        if split == "train":
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return pd.DataFrame(X, index=split_feat.index, columns=split_feat.columns)

    def _attribute_graph(self, graph: Data, feat: pd.DataFrame) -> Data:

        assert feat.isna().any().any() == False

        feat.index = feat.index.droplevel("core").astype(int)
        feat = feat.loc[graph.cell_ids, :]  # align the features with the graph
        graph.x = torch.tensor(feat.values, dtype=torch.float32)
        graph.feature_name = feat.columns.tolist()

        return graph

    def attribute_graphs(
        self, fold_info: pd.DataFrame, feat_dir: Path, output_dir: Path
    ):
        for concept in self.data_config["concepts"]:
            concept_output_dir = output_dir / concept
            concept_output_dir.mkdir(parents=True, exist_ok=True)
            for core in fold_info.index:
                graph_path = self.concept_graphs_dir / concept / f"{core}.pt"
                g = torch.load(graph_path)

                feat_path = feat_dir / f"{core}.parquet"
                feat = pd.read_parquet(feat_path)
                g_attr = self._attribute_graph(g, feat)
                g_attr.y = torch.tensor([fold_info.loc[core, "target"]])

                g_attr.core = core
                g_attr.concept = concept
                torch.save(g_attr, concept_output_dir / f"{core}.pt")

    def process(self):
        sample_data = self.gather_sample_data()
        valid_samples = self.filter_samples(sample_data)
        encoder = LabelEncoder()
        encoded_target = encoder.fit_transform(
            valid_samples[self.data_config["targets"]]
        )
        valid_samples = valid_samples.assign(target=encoded_target)
        folds = self.split_samples_into_folds(valid_samples)

        # iterate over all samples.parquet in folds_dir
        for fold_info_path in self.folds_dir.glob("*/info.parquet"):
            fold_info = pd.read_parquet(fold_info_path)

            feat_dir = fold_info_path.parent / "normalized_features"
            feat_dir.mkdir(parents=True, exist_ok=True)
            self.normalize_features(fold_info, feat_dir)

            output_dir = fold_info_path.parent / "attributed_graphs"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.attribute_graphs(fold_info, feat_dir, output_dir)

    def split_samples_into_folds(self, valid_samples) -> list[pd.DataFrame]:
        func = SPLIT_STRATEGIES[self.data_config["split"]["strategy"]]
        folds = func(valid_samples, **self.data_config["split"]["kwargs"])
        for i, fold in enumerate(folds):
            out_dir = self.folds_dir / f"fold_{i}"
            out_dir.mkdir(parents=True, exist_ok=True)
            fold.to_parquet(out_dir / f"info.parquet")
        return folds

        # split data
        # normalize data

    @property
    def processed_file_names(self) -> list:
        return self._processed_file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(self.root / self.processed_file_names[idx])
