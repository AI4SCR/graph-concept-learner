from typing import List

import pandas as pd
import torch
from pathlib import Path
import os.path as osp
from torch_geometric.data import Dataset
import yaml
from graph_cl.graph_builder.attribute_graph import attribute_graph
from graph_cl.preprocessing.splitV2 import SPLIT_STRATEGIES
from sklearn.preprocessing import LabelEncoder


class ConceptDataset(Dataset):
    def __init__(
        self,
        experiment_dir: Path,
        data_dir: Path,
    ) -> None:
        # super().__init__()

        # paths
        self.experiment_dir = experiment_dir
        self.data_dir = data_dir

        self.concept_graphs_dir = self.data_dir / "03_concept_graphs"
        self.attributed_graphs_dir = (
            self.experiment_dir / "data" / "04_attributed_graphs"
        )
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
        # TODO: how to handle nan strings?
        targets = sample_data[self.data_config["targets"]]
        # valid_cores = sample_data[targets.notna().all(1)]
        valid_cores = sample_data[targets.notna()]
        return valid_cores

    def filter_samples(self, sample_data):
        sample_data = self.sample_has_targets(sample_data)
        m = (
            sample_data[self.data_config["concepts"]]
            >= self.data_config["filter"]["min_cells_per_graph"]
        ).all(1)
        return sample_data[m]

    def attribute_graphs(self, valid_samples: pd.DataFrame):
        for concept in self.data_config["concepts"]:
            for core in valid_samples.index:
                graph_path = self.concept_graphs_dir / concept / f"{core}.pt"
                g = torch.load(graph_path)

                g_attr = attribute_graph(
                    core=core,
                    graph=g,
                    attribute_config=self.attribute_config,
                    processed_dir=self.data_dir / "02_processed",
                )
                g_attr.core = core
                g_attr.concept = concept

                torch.save(g_attr, self.attributed_graphs_dir / concept / f"{core}.pt")

    def process(self):
        sample_data = self.gather_sample_data()
        valid_samples = self.filter_samples(sample_data)
        encoded_target = LabelEncoder().fit_transform(
            valid_samples[self.data_config["targets"]]
        )
        valid_samples = valid_samples.assign(target=encoded_target)
        folds = self.split_samples_into_folds(valid_samples)

        # iterate over all samples.parquet in folds_dir
        # for each fold construct train, val and train, collect feature data from data_dir
        # save the features in 05_fold/fold_i/feature.parquet
        # normalize features and save in 05_fold/fold_i/feature_normalized.parquet
        # attribute graph based on normalized features
        # save graphs to 05_fold/fold_i/attributed_graphs

        # self.attribute_graphs(valid_samples)
        # for fold in folds:
        #     for split in fold.split.unique():
        #         torch.save(fold, self.folds_dir / f"{fold.name}.pt")

    def split_samples_into_folds(self, valid_samples) -> list[pd.DataFrame]:
        func = SPLIT_STRATEGIES[self.data_config["split"]["strategy"]]
        folds = func(valid_samples, **self.data_config["split"]["kwargs"])
        for i, fold in enumerate(folds):
            out_dir = self.folds_dir / f"fold_{i}"
            out_dir.mkdir(parents=True, exist_ok=True)
            fold.to_parquet(out_dir / f"samples.parquet")
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
