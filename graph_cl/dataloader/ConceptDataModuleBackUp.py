import logging

import lightning as L
import pandas as pd
from torch_geometric.data import DataLoader, Dataset
import torch
from pathlib import Path

from graph_cl.preprocessing.split import SPLIT_STRATEGIES
from ..data_models.Data import DataConfig
from sklearn.preprocessing import LabelEncoder

from ..preprocessing.filter import filter_samples
from ..preprocessing.normalize import Normalizer
from ..preprocessing.attribute import collect_features, attribute_graph


class ConceptDataset(Dataset):
    def __init__(self, data):
        self.data = data
        super().__init__()

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class ConceptDataModule(L.LightningDataModule):
    def __init__(
        self,
        samples: pd.DataFrame,
        config: DataConfig,
        concepts: list[str] = None,
        batch_size: int = 8,
        shuffle: bool = True,
        cache_dir: Path = None,
    ):
        super().__init__()

        # init params
        self.samples = samples

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_dir = cache_dir

        self.target_encoder = LabelEncoder()

        self.config = config
        self.target = config.target
        # note: this enables that we can train on single concepts
        self.concepts = concepts if concepts else self.config.concepts

        self.normalize = Normalizer(**self.config.normalize.kwargs)

        import uuid

        root = self.cache_dir / str(uuid.uuid4())
        logging.info(f"Using cache dir: {root}")

        self.info_path = root / "info.parquet"

        self.feat_norm_dir = root / "normalized_features"
        self.feat_norm_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_path = root / "attributed_graphs"

    def prepare_data(self):
        samples = filter_samples(metadata=self.samples, **self.config.filter.dict())
        assert samples.isna().any() == False

        # TODO: it would be better to encode on the train split only
        targets_encoded = self.target_encoder.fit_transform(self.samples.target)
        self.samples.assign(y=targets_encoded)

        func = SPLIT_STRATEGIES[self.config.split.strategy]
        samples = func(self.samples, **self.config.split.kwargs)
        samples.to_parquet(self.cache_dir / "samples.parquet")

        data = collect_features(samples=self.samples, config=self.config.features)
        data.to_parquet(self.cache_dir / "data.parquet")

    def setup(self, stage: str):
        spls = samples = pd.read_parquet(self.cache_dir / "samples.parquet")
        samples = self.samples[self.samples.stage == stage]

        data = None
        split_feat = self.data.loc[spls.index, :]
        assert split_feat.index.get_level_values("cell_id").isna().any() == False

        # note: stage \in {fit,validate,test,predict}
        if stage == "fit":
            feat_norm = self.normalize.fit_transform(split_feat)
        else:
            feat_norm = self.normalize.transform(split_feat)

        graphs = self.attribute_graphs(feat_norm, spls)
        dataset = ConceptDataset(graphs)
        setattr(self, f"{stage}_data", dataset)

    def attribute_graphs(self, feat, samples):
        dataset = []
        for sample_name in samples.index:
            sample = {}
            for concept_name in self.concepts:
                graph_path = samples.loc[sample_name][
                    f"{concept_name}__concept_graph_path"
                ]
                g = torch.load(graph_path)

                g_attr = attribute_graph(g, feat)
                g_attr.y = torch.tensor([samples.loc[sample_name, "y"]])

                g_attr.name = sample_name
                g_attr.concept = concept_name
                sample[concept_name] = g_attr
            dataset.append(sample)
        return dataset

    def train_dataloader(self):
        dataset = getattr(self, "fit_data", None)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        dataset = getattr(self, "val_data", None)
        return DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = getattr(self, "test_data", None)
        return DataLoader(dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        dataset = getattr(self, "predict_data", None)
        return DataLoader(dataset, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
