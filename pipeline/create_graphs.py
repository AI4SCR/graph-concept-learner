import networkx as nx

from graph_cl.graph_builder.knn_graph_builder import KNNGraphBuilder
import yaml
from pathlib import Path
from skimage.io import imread
import torch
from torch_geometric.utils.convert import from_networkx

if __name__ == "__main__":
    concept_config_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/concepts/concept_2_knn.yaml"
    )
    mask_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed/masks/BaselTMA_SP41_100_X15Y5.tiff"
    )
    output_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs/"
    )

    with open(concept_config_path, "r") as f:
        concept_config = yaml.load(f, Loader=yaml.FullLoader)

    mask = imread(mask_path, plugin="tifffile")

    builder = KNNGraphBuilder()
    g = builder.build_graph(mask, **concept_config["graph"]["kwargs"])

    # Remove edge weights
    for (n1, n2, d) in g.edges(data=True):
        d.clear()

    # we set the node attribute to the cell_id to enable tracking after conversion to pyg
    # note: the g.nodes() iteration order should be deterministic, thus it should be nought to
    #   to set pyg.cell_ids = torch.tensor([i for i in g.nodes()])
    #   but for now lets be extra cautious
    import networkx as nx

    nx.set_node_attributes(g, {i: i for i in g.nodes()}, name="cell_id")

    # From netx to pyg
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

    concept_name = concept_config["concept_name"]
    core = mask_path.stem
    output_dir = output_dir / concept_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(pyg, output_dir / f"{core}.pt")
