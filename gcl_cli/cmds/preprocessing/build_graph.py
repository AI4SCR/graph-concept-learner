from skimage.io import imread
import numpy as np
import yaml
from pathlib import Path
import torch
from torch_geometric.utils.convert import from_networkx
from graph_cl.preprocessing.filterV2 import filter_mask_objects
from graph_cl.graph_builder import build_graph
from torch_geometric.data import Data

from graph_cl.configuration.configurator import ConceptConfig


def create_concept_graph(mask: np.ndarray, concept_config: ConceptConfig) -> Data:
    mask = filter_mask_objects(mask, **concept_config["filter"])
    graph = build_graph(
        mask,
        topology=concept_config["topology"],
        builder_kwargs=concept_config["graph"]["kwargs"],
    )

    # Remove edge weights
    for (n1, n2, d) in graph.edges(data=True):
        d.clear()

    # From netx to pyg
    g = from_networkx(G=graph, group_node_attrs=all)

    return g


import click
from . import preprocessing


@preprocessing.command()
@click.argument("mask_path", type=click.Path(exists=True), path_type=Path)
@click.argument("concept_config_path", type=click.Path(exists=True), path_type=Path)
@click.argument("output_dir", type=click.Path(), path_type=Path)
def build_graph(
    mask_path: Path,
    concept_config_path: Path,
    output_dir: Path,
):
    with open(concept_config_path) as f:
        concept_config = yaml.load(f, Loader=yaml.Loader)
        concept_config = ConceptConfig(**concept_config)

    mask = imread(mask_path, plugin="tifffile")
    graph = create_concept_graph(mask=mask, concept_config=concept_config)

    core = mask_path.stem
    concept_name = concept_config["concept_name"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / concept_name / f"{core}.pt"

    torch.save(graph, output_path)
