import lightning as L
import pandas as pd
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from pathlib import Path
import torch

from ..utils.log import logger
from ..data_models.Data import DataConfig

from ..preprocessing.normalize import Normalizer
from ..data_models.Sample import Sample


class ConceptDataset(Dataset):
    def __init__(self, data: list[Data] | list[dict[str, Data]]):
        self.data = data
        super().__init__()

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class ConceptDataModule(L.LightningDataModule):
    def __init__(
        self,
        splits: dict[str, list[Sample]],
        concepts: str | list[str],
        config: DataConfig,
        save_samples_dir: Path,
        save_attributes_dir: Path,
        save_dataset_dir: Path,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        super().__init__()

        # init params
        self.splits = splits
        self.concepts = concepts
        self.config = config
        self.save_samples_dir = save_samples_dir
        self.save_attributes_dir = save_attributes_dir
        self.save_dataset_dir = save_dataset_dir

        self.batch_size = batch_size
        self.shuffle = shuffle

        # computed field
        self.normalize = Normalizer(**self.config.normalize.kwargs)
        self.num_features = self.splits["fit"][0].expression.shape[1]

        # datasets
        self.ds_fit = None
        self.ds_val = None
        self.ds_test = None

    def prepare_data(self):
        # TODO: use the `collect_sample_features` function here to aggregate the features
        feat = pd.concat([s.expression for s in self.splits["fit"]])
        self.normalize.fit(feat)
        self.num_features = feat.shape[1]

        for stage in ["fit", "val", "test", "predict"]:
            if stage not in self.splits:
                continue

            samples = self.splits[stage]
            ds = []
            for s in samples:
                if s.attributes is None:
                    attributes = self.normalize.transform(s.expression)
                    s.attributes_url = (
                        self.save_attributes_dir / stage / f"{s.name}.parquet"
                    )
                    attributes.to_parquet(s.attributes_url)
                    with (self.save_samples_dir / f"{s.name}.json").open(
                        "w", encoding="utf-8"
                    ) as file:
                        file.write(s.model_dump_json(indent=4))

                if isinstance(self.concepts, str):
                    cg = s.attributed_graph(self.concepts)
                else:
                    cg = {
                        concept_name: s.attributed_graph(concept_name)
                        for concept_name in self.concepts
                    }
                ds.append(cg)

            torch.save(ds, self.save_dataset_dir / f"{stage}.pt")
            logger.info(f"saved `{stage}.pt` dataset to {self.save_dataset_dir}")

    def setup(self, stage: str):
        # note: it seems to be necessary to setup all dataloaders or at least the fit, val loaders.
        #   the val_dataloader is called after just running the setup with stage='fit'. Unclear why.

        if stage == "fit":
            self.ds_fit = torch.load(self.save_dataset_dir / f"{stage}.pt")

        if stage == "test":
            self.ds_test = torch.load(self.save_dataset_dir / f"{stage}.pt")

        if stage == "val":
            self.ds_val = torch.load(self.save_dataset_dir / f"{stage}.pt")

    def train_dataloader(self):
        return DataLoader(self.ds_fit, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
