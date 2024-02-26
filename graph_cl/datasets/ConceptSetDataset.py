"""Dataset implementation for a constellation of concept graphs using pytorch geometric"""

import os.path as osp
import torch
from torch_geometric.data import Dataset, InMemoryDataset

from pathlib import Path
import pandas as pd


class ConceptSetDataset(Dataset):
    """
    Implements a data set using pytorch geometric.

    Args:
        root: a directory where all the pytorch geometric graphs are (datatype 'Data').
        The files containing the graphs bust be suffixed '.pt', and inside concept designated
        folder.
    """

    def __init__(
        self,
        root: Path,
        fold_info: pd.DataFrame,
        split: str = "train",
    ) -> None:

        self.root = root
        self.fold_info = fold_info
        self.split = split
        assert split in ["train", "val", "test"]

        self.samples = self.fold_info[self.fold_info.split == self.split].index.tolist()

        self.concept_names = [
            c.name for c in (self.root / "attributed_graphs").glob("*") if c.is_dir()
        ]
        self.num_concepts = len(self.concept_names)

        super().__init__(
            str(self.root), transform=None, pre_transform=None, pre_filter=None
        )

    def len(self):
        return len(self.samples)

    def get(self, idx):
        core = self.samples[idx]
        data_dict = {}

        for concept in self.concept_names:
            concept_graph = torch.load(
                Path(self.root) / "attributed_graphs" / concept / f"{core}.pt"
            )
            data_dict[concept] = concept_graph
        return data_dict


class ConceptSetDatasetMemo(InMemoryDataset):
    """
    Implements a data set using pytorch geometric.

    Args:
        root: a directory where all the pytorch geometric graphs are (datatype 'Data').
        The files containing the graphs bust be suffixed '.pt', and inside concept designated
        folder.
    """

    def __init__(
        self,
        root: Path,
        fold_info: pd.DataFrame,
        split: str = "train",
    ) -> None:

        self.root = root
        self.fold_info = fold_info
        self.split = split
        assert split in ["train", "val", "test"]

        self._len = len(self.fold_info[self.fold_info.split == self.split])

        self.concept_names = [
            c.name for c in (self.root / "attributed_graphs").glob("*") if c.is_dir()
        ]
        self.num_concepts = len(self.concept_names)

        super().__init__(
            str(self.root), transform=None, pre_transform=None, pre_filter=None
        )

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "dataset_gcl")

    @property
    def processed_paths(self) -> list[str]:
        return [osp.join(self.processed_dir, self.split + ".pt")]

    @property
    def processed_file_names(self) -> list:
        return ["train.pt", "test.pt", "val.pt"]

    def process(self):
        for grp_name, grp_cores in self.fold_info.groupby("split"):
            data_dict = {
                concept: [] for concept in self.concept_names
            }  # dict of concepts, with list of Data objects for each core
            for core in grp_cores.index:
                for concept in self.concept_names:
                    concept_graph = torch.load(
                        Path(self.root) / "attributed_graphs" / concept / f"{core}.pt"
                    )
                    data_dict[concept].append(concept_graph)

                    # for key, value in ['edge_index', 'x', 'cell_ids']:
                    #     data[f"{concept}__{key}"] = value
                    #
                    # for key in ['y', 'feature_name', 'core', 'num_nodes', 'concept']:
                    #     if key in concept_graph:
                    #         data[key].append(concept_graph[key])
                    #     else:
                    #         data[key] = concept_graph[key]

            torch.save(data_dict, osp.join(self.processed_dir, f"{grp_name}.pt"))

    def len(self):
        return self._len

    def get(self, idx):
        data_dict = {concept: [] for concept in self.concept_names}
        return {key: value[idx] for key, value in data_dict.items()}
