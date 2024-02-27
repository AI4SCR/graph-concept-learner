import logging

from skimage.io import imread
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import torch
from torch_geometric.utils.convert import from_networkx
from graph_cl.preprocessing.filter import filter_mask_objects
from graph_cl.graph_builder import build_graph
from torch_geometric.data import Data
import networkx as nx
from graph_cl.configuration.configurator import ConceptConfig
import click


def _build_graph(mask: np.ndarray, topology: str, params: dict) -> Data:
    # if there are no objects in the mask, return an empty graph
    if not mask[mask != 0].size:
        return Data()

    graph = build_graph(
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


@click.command()
@click.argument("mask_path", type=click.Path(exists=True, path_type=Path))
@click.argument("labels_path", type=click.Path(exists=True, path_type=Path))
@click.argument("concept_config_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
def build_concept_graph(
    mask_path: Path,
    labels_path: Path,
    concept_config_path: Path,
    output_dir: Path,
):
    logging.info(f"Building graph for {mask_path.stem}")

    with open(concept_config_path) as f:
        concept_config = yaml.load(f, Loader=yaml.Loader)
        concept_config = ConceptConfig(**concept_config)

    core = mask_path.stem
    concept_name = concept_config.concept_name
    output_path = output_dir / concept_name / f"{core}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logging.info(f"Graph for {mask_path.stem} already exists.")
        return

    labels = pd.read_parquet(labels_path)
    labels = labels[concept_config.filter.col_name]

    mask = imread(mask_path, plugin="tifffile")
    mask = filter_mask_objects(
        mask, labels=labels, include_labels=concept_config.filter.include_labels
    )
    graph = _build_graph(
        mask=mask,
        topology=concept_config.graph.topology,
        params=concept_config.graph.params,
    )

    torch.save(graph, output_path)
