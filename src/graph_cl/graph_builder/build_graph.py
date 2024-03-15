import networkx as nx
import numpy as np

from .ContactGraphBuilder import ContactGraphBuilder
from .KNNGraphBuilder import KNNGraphBuilder
from .RadiusGraphBuilder import RadiusGraphBuilder

GRAPH_BUILDERS = {
    "knn": KNNGraphBuilder,
    "contact": ContactGraphBuilder,
    "radius": RadiusGraphBuilder,
}


def build_nx_graph(mask: np.ndarray, topology: str, params: dict) -> nx.Graph:
    """Build graph from mask using specified topology."""

    # Raise error is the builder_type is invalid
    if topology not in GRAPH_BUILDERS:
        raise ValueError(
            f"invalid graph topology {topology}. Available topologies are {GRAPH_BUILDERS.keys()}"
        )

    # Instantiate graph builder object
    builder = GRAPH_BUILDERS[topology]()

    # Build graph and get key
    g = builder.build_graph(mask, **params)
    return g


from torch_geometric.data import Data
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import from_networkx


def build_pyg_graph(mask: np.ndarray, topology: str, params: dict) -> Data:
    # if there are no objects in the mask, return an empty graph
    if not mask[mask != 0].size:
        return Data()

    graph = build_nx_graph(
        mask,
        topology=topology,
        params=params,
    )

    # add object ides as node features
    # note: if we convert to a pyg graph, the node ids will be sequential and we will lose the explicit object ids
    nx.set_node_attributes(graph, {i: i for i in graph.nodes}, "object_id")

    # Remove edge weights
    for (n1, n2, d) in graph.edges(data=True):
        d.clear()

    # From netx to pyg
    g = from_networkx(
        G=graph, group_node_attrs=all
    )  # this fails without nodes attributes
    g.object_id = g.x.flatten()
    g.num_nodes = g.object_id.size(0)
    g.x = None

    return g
