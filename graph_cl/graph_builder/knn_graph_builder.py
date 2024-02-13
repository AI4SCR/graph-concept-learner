from sklearn.neighbors import kneighbors_graph
import networkx as nx
import pandas as pd
import numpy as np
from .base_graph_builder import BaseGraphBuilder


class KNNGraphBuilder(BaseGraphBuilder):
    """KNN (K-Nearest Neighbors) class for graph building."""

    def build_graph(self, mask: np.ndarray, **kwargs):
        # Extract location:
        ndata = self.object_coordinates(mask)

        adj = kneighbors_graph(ndata.to_numpy(), **kwargs)
        df = pd.DataFrame(adj.A, index=ndata.index, columns=ndata.index)
        self.graph = nx.from_pandas_adjacency(df)

        return self.graph
