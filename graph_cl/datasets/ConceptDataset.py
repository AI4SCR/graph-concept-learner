from typing import List

import pandas as pd
import torch
from pathlib import Path
import os.path as osp
from torch_geometric.data import InMemoryDataset


class CptDatasetMemo(InMemoryDataset):
    def __init__(
        self,
        root: Path,
        fold_info: pd.DataFrame,
        concept: str,
        split: str = "train",
    ) -> None:

        self.root = root
        self.fold_info = fold_info
        self.concept = concept
        self.split = split
        assert split in ["train", "val", "test"]

        super().__init__(
            str(self.root), transform=None, pre_transform=None, pre_filter=None
        )
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return str(self.root)

    @property
    def processed_paths(self) -> List[str]:
        return [osp.join(self.root, self.split + ".pt")]

    @property
    def processed_file_names(self) -> list:
        return ["train.pt", "test.pt", "val.pt"]

    def process(self):
        for grp_name, grp_cores in self.fold_info.groupby("split"):
            data_list = [
                torch.load(
                    Path(self.root) / "attributed_graphs" / self.concept / f"{core}.pt"
                )
                for core in grp_cores.index
            ]
            self.save(data_list, osp.join(self.processed_dir, f"{grp_name}.pt"))
