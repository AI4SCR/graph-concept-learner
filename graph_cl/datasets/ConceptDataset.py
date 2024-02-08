from typing import List

import pandas as pd
import torch
from pathlib import Path
import os.path as osp
from torch_geometric.data import Dataset, InMemoryDataset


class ConceptDataset(Dataset):
    def __init__(
        self,
        root: Path,
        fold_meta_data: pd.DataFrame,
        split: str = "train",
    ) -> None:
        super().__init__(str(root), transform=None, pre_transform=None, pre_filter=None)

        assert root.exists()
        self.root = root

        self.fold_meta_data = fold_meta_data

        assert split in ["train", "val", "test"]
        self.split = split

        cores = self.fold_meta_data[self.fold_meta_data["split"] == self.split]["core"]
        self._processed_file_names = [f"{core}.pt" for core in cores]

    @property
    def processed_file_names(self) -> list:
        return self._processed_file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(self.root / self.processed_file_names[idx])


class CptDatasetMemo(InMemoryDataset):
    def __init__(
        self,
        root: Path,
        fold_meta_data: pd.DataFrame,
        split: str = "train",
    ) -> None:

        assert root.exists()

        self.fold_meta_data = fold_meta_data

        assert split in ["train", "val", "test"]
        self.split = split

        # cores = self.fold_meta_data[self.fold_meta_data['split'] == self.split]['core']
        self._raw_file_names = [f"{core}.pt" for core in self.fold_meta_data.core]

        super().__init__(str(root), transform=None, pre_transform=None, pre_filter=None)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_paths(self) -> List[str]:
        return [osp.join(self.processed_dir, self.split + ".pt")]

    @property
    def raw_file_names(self) -> list:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> list:
        return ["train.pt", "test.pt", "val.pt"]

    def process(self):
        for grp_name, grp_cores in self.fold_meta_data.groupby("split"):
            data_list = [torch.load(Path(self.root) / f) for f in grp_cores.core]
            # data_list = [torch.load(Path(self.root) / f) for f in ['BaselTMA_SP41_41_X11Y8.pt', 'BaselTMA_SP42_152_X15Y6.pt']]
            self.save(data_list, osp.join(self.processed_dir, f"{grp_name}.pt"))
