"""Dataset implementation for concept graphs using pytorch geometric"""

import os.path as osp
import os
import torch
from torch_geometric.data import Dataset, Data


class Concept_Dataset(Dataset):
    """
    Implements a data set using pytorch geometric.

    Args:
        root: a directory where all the pytorch geometric graphs are (datatype 'Data').
        The files containing the graphs bust be suffixed '.pt'.
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        # list to store file names
        self.file_names = []

        # Save name of the files in root
        for file in os.listdir(self.root):
            # check only text files
            if file.endswith(".pt"):
                self.file_names.append(file)

        # sort alphabetically
        self.file_names.sort()

    def len(self):
        """
        Method to get the number of observation is the concept graph dataset.

        Returns:
            Number of observations in the dataset.
        """
        if self._indices is None:
            return len(self.file_names)
        else:
            return len(self._indices)

    def get(self, idx: int) -> Data:
        """
        Method to load the i'th observation of the dataset.

        It could happen that after splinting some datums/indexes
        are not longe accessible with this method, since the method
        fetches data based on their absolute index (order of files in
        `self.file_names`) which is subseted when a dataset is split.

        Subscript operator `[]` invokes `__getitem__()`.
        When the `self._indices` variable is not `None`
        (when `ConceptDataset` has been splitted) `[i]`
        returns the i'th index from the `self._indices` list.
        Therefore `i` in the context of the subscript operator references
        a position of the data in a dataset rather than its absolute index.

        Args:
            idx: The index of the observation as in self.file_names list.

        Returns:
            An instance of torch_geometric "Data".
        """
        if self._indices is None:
            return torch.load(osp.join(self.root, self.file_names[idx]))
        else:
            if idx in self._indices:
                return torch.load(osp.join(self.root, self.file_names[idx]))
            else:
                raise KeyError(
                    f"This dataset does not contain a datum with index {idx}.\n"
                    "Printing index: \n"
                    f"{self._indices}"
                )
