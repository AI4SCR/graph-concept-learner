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
from ..preprocessing.filter import collect_metadata, filter_samples
from ..preprocessing.normalize import Normalizer
from ..preprocessing.attribute import collect_features, attribute_graph


class ConceptDataModule(L.LightningDataModule):
    def __init__(
        self,
        labels_dir: Path,
        concept_graphs_dirs: list[Path],
        processed_dir: Path,
        attributed_graphs_dir: Path,
        config: DataConfig,
        batch_size: int = 32,
    ):
        super().__init__()

        # init params
        self.target_encoder = LabelEncoder()
        self.samples = None
        self.metadata = None
        self.labels_dir = labels_dir
        self.concept_graphs_dirs = concept_graphs_dirs
        self.processed_dir = processed_dir
        self.attributed_graphs_dir = attributed_graphs_dir
        self.attributed_graphs_dir.mkdir(parents=True, exist_ok=True)

        # self.feat_norm_dir = fold_info_path.parent / "normalized_features"
        # self.feat_norm_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.target = config.target
        self.concepts = self.config.concepts

        self.normalize = Normalizer(**self.config.normalize.kwargs)

    def prepare_data(self) -> None:
        # contains only information relevant for splitting
        self.samples = collect_metadata(
            target=self.target,
            labels_dir=self.labels_dir,
            concept_graphs_dirs=self.concept_graphs_dirs,
        )
        self.samples = filter_samples(
            metadata=self.metadata, **self.config.filter.dict()
        )
        assert self.samples.isna().any() == False

        targets_encoded = self.target_encoder.fit_transform(self.samples.target)
        self.samples.assign(target=targets_encoded)

        func = SPLIT_STRATEGIES[self.config.split.strategy]
        split_metadata = func(self.samples, **self.config.split.kwargs)

        feat = collect_features(
            processed_dir=self.processed_dir, feat_config=self.config.features
        )

        # feat_norm_dir = fold_info_path.parent / "normalized_features"
        # feat_norm_dir.mkdir(parents=True, exist_ok=True)

        # normalize_features(
        #     feat=feat,
        #     fold_info=fold_info,
        #     output_dir=feat_norm_dir,
        #     method=data_config.normalize.method,
        #     **data_config.normalize.kwargs,
        # )

        # self.batch_size = batch_size

    def save_to_disk(self, data, path: Path):
        data.to_parquet(path)

    def setup(self, stage: str):
        # contains only information relevant for splitting
        self.samples = collect_metadata(
            target=self.target,
            labels_dir=self.labels_dir,
            concept_graphs_dirs=self.concept_graphs_dirs,
        )
        self.samples = filter_samples(
            metadata=self.metadata, **self.config.filter.dict()
        )
        assert self.samples.isna().any() == False

        targets_encoded = self.target_encoder.fit_transform(self.samples.target)
        self.samples.assign(target=targets_encoded)

        func = SPLIT_STRATEGIES[self.config.split.strategy]
        self.samples = func(self.samples, **self.config.split.kwargs)

        feat = collect_features(
            processed_dir=self.processed_dir, feat_config=self.config.features
        )

        spls = self.samples[self.samples.stage == stage]
        split_feat = feat.loc[spls.index, :]
        assert split_feat.index.get_level_values("cell_id").isna().any() == False

        # note: stage \in {fit,validate,test,predict}
        if stage == "fit":
            feat_norm = self.normalize.fit_transform(split_feat)
        else:
            feat_norm = self.normalize.transform(split_feat)

        graphs = self.attribute_graphs(feat_norm, self.samples)
        setattr(self, f"{stage}_data", graphs)

    def attribute_graphs(self, feat, samples):
        dataset = []
        for sample_name in self.samples.index:
            sample = {}
            for concept_graphs_dir in self.concept_graphs_dirs:
                concept = concept_graphs_dir.name
                graph_path = concept_graphs_dir / f"{sample_name}.pt"
                g = torch.load(graph_path)

                g_attr = attribute_graph(g, feat)
                g_attr.y = torch.tensor([samples.loc[sample_name, "target"]])

                g_attr.name = sample_name
                g_attr.concept = concept
                sample[concept] = g_attr
            dataset.append(sample)
        return dataset

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
