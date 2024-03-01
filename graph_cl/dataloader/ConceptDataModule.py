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
        self.labels_dir = labels_dir
        self.concept_graphs_dirs = concept_graphs_dirs
        self.processed_dir = processed_dir
        self.attributed_graphs_dir = attributed_graphs_dir
        self.attributed_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.target = config.target

    def prepare_data(self) -> None:
        # contains only information relevant for splitting
        self.metadata = collect_metadata(
            target=self.target,
            labels_dir=self.labels_dir,
            concept_graphs_dirs=self.concept_graphs_dirs,
        )
        self.samples = filter_samples(
            metadata=self.metadata, **self.config.filter.dict()
        )
        assert self.samples.isna().any() == False

        self.target_encoder = LabelEncoder()
        targets_encoded = self.target_encoder.fit_transform(self.samples.target)
        self.samples.assign(target=targets_encoded)

        func = SPLIT_STRATEGIES[self.config.split.strategy]
        folds = func(self.samples, **self.config.split.kwargs)
        for fold in folds:
            fold.to_parquet("PATH_TO_DATALOADER")
        # self.batch_size = batch_size

    def save_to_disk(self, data, path: Path):
        data.to_parquet(path)

    def setup(self, stage: str):

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
