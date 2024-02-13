import logging

import numpy as np
import pandas as pd

from graph_cl.graph_builder.graphBuilder import build_graph
import yaml
from pathlib import Path
from skimage.io import imread
import torch
from torch_geometric.utils.convert import from_networkx

logging.basicConfig(level=logging.INFO)


def filter_masks(
    mask: np.ndarray, obs_labels: pd.DataFrame, col_name: str, include_labels: list[str]
):
    mask = mask.copy()
    m = obs_labels[col_name].isin(include_labels)
    remove_objs = set(obs_labels.loc[~m].index)
    for obj_id in remove_objs:
        mask[mask == obj_id] = 0
    return mask


if __name__ == "__main__":
    force_load = False
    masks_dir_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed/masks"
    )
    obs_labs_dir_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed/labels/observations"
    )
    output_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs/"
    )
    concept_config_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/00_concepts"
    )

    for obs_labs_path in obs_labs_dir_path.glob("*.parquet"):
        core = obs_labs_path.stem
        logging.info(f"Processing {core}")

        mask_path = masks_dir_path / f"{core}.tiff"
        mask = imread(mask_path, plugin="tifffile")

        for concept_config_path in concept_config_dir.glob("*.yaml"):
            with open(concept_config_path, "r") as f:
                concept_config = yaml.load(f, Loader=yaml.FullLoader)
            concept_name = concept_config["concept_name"]
            logging.info(f"Generate {concept_name}")

            concept_output_dir = output_dir / concept_name
            concept_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = concept_output_dir / f"{core}.pt"
            if output_path.exists() and not force_load:
                continue

            obs_labels = pd.read_parquet(obs_labs_path)
            concept_mask = filter_masks(mask, obs_labels, **concept_config["filter"])

            g = build_graph(concept_mask, **concept_config["graph"])

            # Remove edge weights
            for (n1, n2, d) in g.edges(data=True):
                d.clear()

            # we set the node attribute to the cell_id to enable tracking after conversion to pyg
            # note: the g.nodes() iteration order should be deterministic, thus it should be nought to
            #   to set pyg.cell_ids = torch.tensor([i for i in g.nodes()])
            #   but for now lets be extra cautious
            import networkx as nx

            # NOTE: we have to cast to int because otherwise cast to tensor might fail due to numpy casting to uint16
            #   this issue seems to depend on the order of the nodes in the graph.
            nx.set_node_attributes(g, {i: i for i in g.nodes()}, name="cell_id")

            # From netx to pyg
            # NOTE: if conversion fails, check that attrs can be casted to tensors.
            #  For example the g.nodes() dtype cannot be uint16 as this type does not exist in torch
            pyg = from_networkx(G=g, group_node_attrs=all)
            pyg.cell_ids = pyg.x.flatten()
            pyg.x = None
            pyg.num_nodes = len(g)

            # alternative, more concise way to set cell_ids
            # pyg = from_networkx(G=g)
            # pyg.cell_ids = torch.tensor([i for i in g.nodes()])

            # check that neighbors are correct after conversion
            # idx_of_obj_1 = int((pyg.x == 1).flatten().nonzero())
            # edge_idcs = (pyg.edge_index[0] == idx_of_obj_1).nonzero().flatten()
            # neighbors = pyg.edge_index[1, edge_idcs]
            # neigh_obj_ids = pyg.x[neighbors]
            # list(nx.neighbors(g, 1))

            torch.save(pyg, output_path)
