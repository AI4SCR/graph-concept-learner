import numpy as np

from .contact_graph_builder import ContactGraphBuilder
from .knn_graph_builder import KNNGraphBuilder
from .radius_graph_builder import RadiusGraphBuilder

GRAPH_BUILDERS = {
    "knn": KNNGraphBuilder,
    "contact": ContactGraphBuilder,
    "radius": RadiusGraphBuilder,
}


def build_graph(mask: np.ndarray, topology: str, builder_kwargs: dict):
    """Build graph from mask using specified topology."""

    # Raise error is the builder_type is invalid
    if topology not in GRAPH_BUILDERS:
        raise ValueError(
            f"invalid graph topology {topology}. Available topologies are {GRAPH_BUILDERS.keys()}"
        )

    # Instantiate graph builder object
    builder = GRAPH_BUILDERS[topology]()

    # Build graph and get key
    g = builder.build_graph(mask, **builder_kwargs)
