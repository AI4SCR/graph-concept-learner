"""Dataset implementation for a constellation of concept graphs using pytorch geometric"""

import os.path as osp
import os
import torch
from torch_geometric.data import Dataset, Data


class ConceptSetDatum(Data):
    """
    One data point in a ConceptSetDataset dataset. This is necessary to ensure correct batching.
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
        The files containing the graphs bust be suffixed '.pt', and inside concept designated
        folder.
    """

    def __init__(
        self, config: dict, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(config, transform, pre_transform, pre_filter)

        # Get concept names
        self.concept_names = list(config.keys())

        # Number of concepts
        self.num_concepts = len(config)

        # Dictionary key=concept_name and value=path_to dir_with_graphs
        self.concept_dict = {key: value["data"] for key, value in config.items()}

        # list to store file names
        self.file_names = []

        # Save name of the files (all concepts should have the same)
        for file in os.listdir(config[self.concept_names[0]]["data"]):
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

    def get(self, idx: int) -> ConceptSetDatum:
        """
        Method to load the i'th observation of the dataset.

        It could happen that after splinting some datums/indexes
        are not longe accessible with this method, since the method
        fetches data based on their absolute index (order of files in
        `self.file_names`) which is subseted when a dataset is split.

        Subscript operator `[]` invokes `__getitem__()`.
        When the `self._indices` variable is not `None`
        (when `ConceptSetDataset` has been splitted) `[i]`
        returns the i'th index from the `self._indices` list.
        Therefore `i` in the context of the subscript operator references
        a position of the data in a dataset rather than its absolute index.

        Args:
            idx: The index of the observation as in self.file_names list.

        Returns:
            An instance of torch_geometric "Data".
        """

        if self._indices is None:
            return self._get(idx)
        else:
            if idx in self._indices:
                return self._get(idx)
            else:
                raise KeyError(
                    f"This dataset does not contain a datum with index {idx}.\n"
                    "Printing index: \n"
                    f"{self._indices}"
                )

    def _get(self, idx: int) -> ConceptSetDatum:
        datum = ConceptSetDatum()
        for concept_name, concept_dir in self.concept_dict.items():
            data = torch.load(osp.join(concept_dir, self.file_names[idx]))
            setattr(datum, f"{concept_name}__x", data.x)
            setattr(datum, f"{concept_name}__edge_index", data.edge_index)

        setattr(datum, "y", data.y)
        setattr(datum, "sample_id", self.file_names[idx].split(".")[0])
        return datum
