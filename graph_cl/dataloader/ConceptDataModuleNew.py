import lightning as L
import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from ..data_models.Data import DataConfig

from ..preprocessing.normalize import Normalizer
from ..data_models.Sample import Sample


class ConceptDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples
        super().__init__()

    def len(self):
        return len(self.samples)

    def get(self, idx):
        return self.samples[idx]


class ConceptDataModule(L.LightningDataModule):
    def __init__(
        self,
        samples: list[Sample],
        concepts: str | list[str],
        config: DataConfig,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        super().__init__()

        # init params
        self.concepts = concepts
        self.samples = samples
        self.config = config

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.normalize = Normalizer(**self.config.normalize.kwargs)

        # TODO: use the `collect_sample_features` function here to aggregate the features
        feat = pd.concat([s.expression for s in self.splits["fit"]])
        self.normalize.fit(feat)  # we fit the normalizer on the training data once

        self.num_features = feat.shape[1]

    def setup(self, stage: str):
        # note: it seems to be necessary to setup all dataloaders or at least the fit, val loaders.
        #   the val_dataloader is called after just running the setup with stage='fit'. Unclear why.
        for _stage in ["fit", "val", "test"]:
            if hasattr(self, f"{_stage}_data"):
                continue

            samples = self.splits[_stage]
            ds = []
            for s in samples:
                assert s.split == _stage
                assert (
                    s.attributes is None
                )  # could it happen that we attribute multiple times, except for running fit/test multiple times?
                s.attributes = self.normalize.transform(s.expression)

                if isinstance(self.concepts, str):
                    cg = s.attributed_graph(self.concepts)
                else:
                    cg = {
                        s.attributed_graph(concept_name)
                        for concept_name in self.concepts
                    }
                ds.append(cg)

            dataset = ConceptDataset(ds)
            setattr(self, f"{_stage}_data", dataset)

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
