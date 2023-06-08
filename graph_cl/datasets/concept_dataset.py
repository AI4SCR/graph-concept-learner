"""Dataset implementation for conceopt graphs using pytorch geometric"""

import os.path as osp
import os
import torch
from torch_geometric.data import Dataset


class Concept_Dataset(Dataset):
    """
    Implements a data set using pytorch geometric.

    Args:
        root: a directory where all the pytorch geometric graphs are (datatype 'Data').
        The files containg the graphs bust be suffixed '.pt'.
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
        return len(self.file_names)

    def get(self, idx: int):
        """
        Method to load the i'th observation of the dataset.

        Args:
            idx: The index of the observation as in self.file_names list.

        Returns:
            An isntacne of toch_geometric "Data".
        """
        data = torch.load(osp.join(self.root, self.file_names[idx]))
        return data
