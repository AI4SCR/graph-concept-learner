from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
import pandas as pd
import numpy as np
from .base_graph_builder import BaseGraphBuilder


class RadiusGraphBuilder(BaseGraphBuilder):
    """
    Radius graph class for graph building.
    """

    def build_graph(self, mask: np.ndarray, **kwargs):
        ndata = self.object_coordinates(mask)
        if len(ndata) == 0:
            return nx.Graph()

        adj = radius_neighbors_graph(ndata.to_numpy(), **kwargs)
        df = pd.DataFrame(adj.A, index=ndata.index, columns=ndata.index)
        self.graph = nx.from_pandas_adjacency(
            df
        )  # this does not add the nodes in the same sequence as the index, column

        return self.graph
