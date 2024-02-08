import pickle

import pandas as pd
import yaml
from pathlib import Path
import athena as ath
import torch
from torch_geometric.utils.convert import from_networkx


def create_graph(
    so, core: str, concept_config: dict, attribut_config: dict, target: int
):

    concept_config["build_and_attribute"] = True

    # merge configs
    config = {}
    config.update(concept_config)
    config.update(attribut_config)

    X = so.X[core]
    X.index.value_counts()

    concept_name = concept_config["concept_name"]
    ath.graph.build_graph(so, core, config=config, key_added=concept_name)

    # Remove edge weights
    for (n1, n2, d) in so.G[core][concept_name].edges(data=True):
        d.clear()

    # From netx to pyg
    g = from_networkx(G=so.G[core][concept_name], group_node_attrs=all)

    # Attach label
    g.y = torch.tensor([target])

    return g


def create_graph_from_files(
    so_path: Path,
    core: str,
    concept_config_path: Path,
    attribute_config_path: Path,
    valid_cores_path: Path,
    output_dir: Path,
):
    # NOTE: It is highly inefficient to load the same so.pkl file for each core
    with open(so_path, "rb") as f:
        so = pickle.load(f)

    with open(concept_config_path) as f:
        concept_config = yaml.load(f, Loader=yaml.Loader)
        concept_name = concept_config["concept_name"]

    with open(attribute_config_path) as f:
        attribut_config = yaml.load(f, Loader=yaml.Loader)

    valid_cores = pd.read_csv(valid_cores_path, index_col="core")

    target = valid_cores.loc[core, "target"]
    graph = create_graph(so, core, concept_config, attribut_config, target)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{core}_{concept_name}.pt"
    torch.save(graph, output_path)


def create_graphs_from_files(
    so,
    core: str,
    concept_config_dir: Path,
    attribute_config_path: Path,
    valid_cores_path: Path,
    output_dir: Path,
):
    for config_path in concept_config_dir.glob("*.yaml"):
        with open(config_path) as f:
            concept_config = yaml.load(f, Loader=yaml.Loader)
            concept_name = concept_config["concept_name"]

        with open(attribute_config_path) as f:
            attribute_config = yaml.load(f, Loader=yaml.Loader)

        valid_cores = pd.read_csv(valid_cores_path, index_col="core")

        target = valid_cores.loc[core, "target"]
        graph = create_graph(so, core, concept_config, attribute_config, target)

        output_path = output_dir / concept_name / f"{core}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, output_path)
