import numpy as np
from skimage.morphology import binary_dilation
from .base_graph_builder import BaseGraphBuilder
import networkx as nx
from tqdm import tqdm

from skimage.morphology import square, diamond, disk

DILATION_KERNELS = {"disk": disk, "square": square, "diamond": diamond}

EDGE_WEIGHT = "weight"


class ContactGraphBuilder(BaseGraphBuilder):
    """Contact-Graph class.

    Build contact graph based on pixel expansion of cell masks.
    """

    def build_graph(self, mask: np.ndarray, **kwargs) -> nx.Graph:
        """Build topology using pixel expansion of segmentation masks provided by topo_data['mask']. Masks that overlap after expansion are connected in the graph."""

        # instantiate dilation kernel
        if kwargs["dilation_kernel"] in DILATION_KERNELS:
            kernel = DILATION_KERNELS[kwargs["dilation_kernel"]](kwargs["radius"])
        else:
            raise ValueError(
                f'Specified dilate kernel not available. Please use one of {{{", ".join(DILATION_KERNELS)}}}.'
            )

        objs = np.unique(mask)
        objs = objs[objs != 0]  # remove background

        # Add nodes to graph
        self.graph.add_nodes_from(objs)

        # compute neighbors (object = the mask of a single cell)
        edges = []
        for obj in tqdm(objs):
            # This creates the augmented object mask in a bool array from
            dilated_img = binary_dilation(mask == obj, kernel)

            # TODO: could be better with set operations, set.remove()
            cells = np.unique(
                mask[dilated_img]
            )  # This identifies the intersecting objects
            cells = cells[cells != obj]  # remove object itself
            cells = cells[cells != 0]  # remove background

            # Appends a list of the edges found at this iteration
            edges.extend([(obj, cell, {EDGE_WEIGHT: 1}) for cell in cells])

        # Adds edges to instance variable graph object
        self.graph.add_edges_from(edges)

        # include self edges
        if kwargs.get("include_self", False):
            edge_list = [(i, i) for i in self.graph.nodes]
            self.graph.add_edges_from(edge_list)

        return self.graph
