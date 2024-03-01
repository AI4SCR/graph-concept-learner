import lightning as L
from torch_geometric.data import DataLoader
from pathlib import Path
import pandas as pd
import logging

from sklearn.preprocessing import LabelEncoder

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


class ConceptDataModule(L.LightningDataModule):
    def __init__(
        self,
        concept_graphs_dir: Path,
        attributed_graphs_dir: Path,
        target: str,
        batch_size: int = 32,
    ):
        super().__init__()

        self.concept_graphs_dir = concept_graphs_dir
        self.attributed_graphs_dir = attributed_graphs_dir
        self.attributed_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.target = target
        self.target_encoder = LabelEncoder()

        self.batch_size = batch_size

    def collect_metadata(self) -> pd.DataFrame:
        metadata = pd.DataFrame()
        for graph_path in self.concept_graphs_dir.glob("*.pt"):
            core = graph_path.stem
            g = torch.load(graph_path)
            metadata.loc[core, "num_nodes"] = g.num_nodes
        return metadata

    def filter_samples(
        self, sample_data, concepts: list[str], target: str, min_cells_per_graph: int
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

    def save_to_disk(self, data, path: Path):
        data.to_parquet(path)

    def setup(self, stage: str):
        labels = pd.read_parquet(self.data_dir / "02_processed" / "labels" / "samples")
        labels = labels[[self.target, "cohort"]].rename(columns={self.target: "target"})

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

        if stage == "fit":
            target_encoded = self.target_encoder.fit_transform(labels.target)
            labels.assign(target=target_encoded)
        else:
            target_encoded = self.target_encoder.transform(labels.target)
            labels.assign(target=target_encoded)

        # for each sample in split
        # create concept graph for sample
        #
        #
        pass

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
