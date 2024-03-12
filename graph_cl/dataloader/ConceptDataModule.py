import lightning as L
import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch
from pathlib import Path

from ..data_models.Data import DataConfig

from ..preprocessing.normalize import Normalizer
from ..preprocessing.attribute import attribute_graph


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
        features: pd.DataFrame,
        config: DataConfig,
        concepts: list[str] = None,
        batch_size: int = 8,
        shuffle: bool = True,
        is_set: bool = False,
        cache_dir: Path = None,
    ):
        super().__init__()

        # init params
        self.samples = samples
        self.features = features
        self.config = config
        # note: this enables that we can train on single concepts
        self.concepts = concepts if concepts else self.config.concepts
        self.is_set = is_set

        self.batch_size = batch_size
        self.shuffle = shuffle
        # self.cache_dir = cache_dir

        self.normalize = Normalizer(**self.config.normalize.kwargs)

        # self.target_encoder = LabelEncoder()
        # self.target = config.target

        # import uuid
        # root = self.cache_dir / str(uuid.uuid4())
        # logging.info(f"Using cache dir: {root}")
        #
        # self.info_path = root / "info.parquet"
        #
        # self.feat_norm_dir = root / "normalized_features"
        # self.feat_norm_dir.mkdir(parents=True, exist_ok=True)
        #
        # self.dataset_path = root / "attributed_graphs"

    def setup(self, stage: str):
        # note: it seems to be necessary to setup all dataloaders or at least the fit, val loaders.
        #   the val_dataloader is call after just running the setup with stage='fit'.
        for _stage in ["fit", "val", "test"]:
            if hasattr(self, f"{_stage}_data"):
                continue
            samples = self.samples[self.samples.stage == _stage]
            split_feat = self.features.loc[samples.index, :]
            assert split_feat.index.get_level_values("cell_id").isna().any() == False

            # note: stage \in {fit,validate,test,predict}
            if _stage == "fit":
                feat_norm = self.normalize.fit_transform(split_feat)
            else:
                feat_norm = self.normalize.transform(split_feat)

            graphs = self.attribute_graphs(feat_norm, samples)
            # note: better way to do this?
            #   we extract the single `Data`, i.e. the graph, from the `sample` if we have a single concept
            if not self.is_set:
                graphs = [g[self.concepts[0]] for g in graphs]
            dataset = ConceptDataset(graphs)
            setattr(self, f"{_stage}_data", dataset)

    def attribute_graphs(self, feat, samples):
        dataset = []
        for sample_name in samples.index:
            sample = {}
            for concept_name in self.concepts:
                graph_path = samples.loc[sample_name][f"{concept_name}__graph_path"]
                g = torch.load(graph_path)

                g_attr = attribute_graph(g, feat.loc[sample_name])
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
