from pathlib import Path

import lightning as L
import torch
from ai4bmr_core.log.log import logger
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

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
        datasets_dir: Path,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        super().__init__()

        # init params
        self.splits = splits
        self.datasets_dir = datasets_dir
        self.concepts = concepts

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_features = self.splits["fit"][0].expression.shape[1]
        self.num_classes = len({s.target for s in self.splits["fit"]})

        if "test" in self.splits:
            # note: ensure that `test` is a subset of `fit`.
            #     maybe, even testing for equality is a good idea.
            assert {s.target for s in self.splits["test"]} <= {
                s.target for s in self.splits["fit"]
            }

        # datasets
        self.ds_fit = None
        self.ds_val = None
        self.ds_test = None

    def prepare_data(self) -> None:
        import shutil

        shutil.rmtree(self.datasets_dir, ignore_errors=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        for stage in ["fit", "val", "test", "predict"]:
            if stage not in self.splits:
                continue
            logger.info(f"Create dataset for stage `{stage}`.")

            samples = self.splits[stage]
            ds = []
            for s in samples:
                # enforce that samples are labelled with the correct stage they belong to
                assert s.stage == stage

                if isinstance(self.concepts, str):
                    cg = s.get_attributed_graph(self.concepts)
                else:
                    cg = {
                        concept_name: s.get_attributed_graph(concept_name)
                        for concept_name in self.concepts
                    }
                ds.append(cg)

            torch.save(ds, self.datasets_dir / f"{stage}.pt")
            logger.debug(f"Dataset saved to {self.datasets_dir}")

    def setup(self, stage: str):
        # note: it seems to be necessary to setup all dataloaders or at least the fit, val loaders.
        #   the val_dataloader is called after just running the setup with stage='fit'. Unclear why.
        #   -> probably because if we `fit` we also have to validate, thus lightning forces you to setup both.
        #   I would prefer a syntax like this, but the call order is setup('fit') -> val_dataloader('val')
        # if stage == "fit":
        #     self.ds_fit = torch.load(self.datasets_dir / f"fit.pt")

        if "fit" in self.splits:
            path_save = self.datasets_dir / f"fit.pt"
            self.ds_fit = torch.load(path_save)
        if "val" in self.splits or "validate" in self.splits:
            path_save = self.datasets_dir / f"val.pt"
            self.ds_val = torch.load(path_save)
        if "test" in self.splits:
            path_save = self.datasets_dir / f"test.pt"
            self.ds_test = torch.load(path_save)

    def train_dataloader(self):
        return DataLoader(self.ds_fit, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
