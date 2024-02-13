from skimage.io import imread
import numpy as np
import yaml
from pathlib import Path
import torch
from torch_geometric.utils.convert import from_networkx
from graph_cl.preprocessing.filterV2 import filter_mask_objects
from graph_cl.graph_builder import build_graph
from torch_geometric.data import Data


def create_concept_graph(mask: np.ndarray, concept_config: dict) -> Data:
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


def create_concept_graph_from_files(
    mask_path: Path,
    concept_config_path: Path,
    output_dir: Path,
):
    with open(concept_config_path) as f:
        concept_config = yaml.load(f, Loader=yaml.Loader)

    mask = imread(mask_path, plugin="tifffile")
    graph = create_concept_graph(mask=mask, concept_config=concept_config)

    core = mask_path.stem
    concept_name = concept_config["concept_name"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / concept_name / f"{core}.pt"

    torch.save(graph, output_path)
