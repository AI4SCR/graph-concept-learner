# %%
"""Dataset implementation for a constellation of conceopt graphs using pytorch geometric"""

import os.path as osp
import os
import torch
from torch_geometric.data import Dataset, Data


class ConceptSetDatum(Data):
    """
    One data point in a Paradigm dataset. This is necessary to ensure correct batching.
    """

    def __init__(self):
        super().__init__()

    def __inc__(self, key, value, *args, **kwargs):
        if key.endswith("edge_index"):
            concept_name = key.split("__")[0]
            x = getattr(self, f"{concept_name}__x")
            return x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ConceptSetDataset(Dataset):
    """
    Implements a data set using pytorch geometric.

    Args:
        root: a directory where all the pytorch geometric graphs are (datatype 'Data').
        The files containg the graphs bust be suffixed '.pt', and inside concept designated
        folder.
    """

    def __init__(
        self, root, exclude, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, exclude, transform, pre_transform, pre_filter)

        # Get concept names
        # Its necesarry to create a copy.
        # No looping and modifying at the same time.
        # Leads to undefined behaviour
        files_in_root = os.listdir(self.root)
        self.concepts_names = files_in_root.copy()

        for x in files_in_root:
            # Remove folders that start with "."
            if x.startswith("."):
                self.concepts_names.remove(x)

            # Remove concepts specified in exclude
            if x in exclude:
                self.concepts_names.remove(x)

        # Add full adress
        self.concept_dirs = [osp.join(self.root, x) for x in self.concepts_names]

        # Create dictionary of concept names and directories
        self.concept_dict = dict(zip(self.concepts_names, self.concept_dirs))

        # Save number of concept names
        self.num_concepts = len(self.concepts_names)

        # list to store file names
        self.file_names = []

        # Save name of the files in root
        for file in os.listdir(osp.join(self.concept_dirs[0])):
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
        datum = ConceptSetDatum()

        for concept_name, concept_dir in self.concept_dict.items():
            data = torch.load(osp.join(concept_dir, self.file_names[idx]))
            setattr(datum, f"{concept_name}__x", data.x)
            setattr(datum, f"{concept_name}__edge_index", data.edge_index)

        setattr(datum, f"y", data.y)
        setattr(datum, f"sample_id", self.file_names[idx].split(".")[0])
        return datum
