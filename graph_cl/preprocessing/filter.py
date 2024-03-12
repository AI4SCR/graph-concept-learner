from pathlib import Path
import torch

import numpy as np
import pandas as pd

from graph_cl.data_models.Sample import Sample


def collect_sample_metadata(sample: Sample, target: str):
    metadata = sample.metadata[[target, "cohort"]]
    metadata.index.name = "sample"
    metadata = metadata.rename(columns={target: "target"})
    concept_graphs = sample.concept_graphs

    for concept_graph_name, concept_graph in concept_graphs:
        metadata.loc[
            sample, f"{concept_graph_name}__num_nodes"
        ] = concept_graph.num_nodes

    return metadata


def collect_metadata(samples: list[Sample], target: str) -> pd.DataFrame:
    """Collect metadata from samples relevant for filtering"""
    cont = [collect_sample_metadata(sample, target) for sample in samples]
    metadata = pd.concat(cont)
    return metadata


def _collect_metadata(
    target: str, labels_dir: Path, concept_graphs_dirs: list[Path]
) -> pd.DataFrame:
    """Collect metadata from samples relevant for filtering"""

    metadata = pd.read_parquet(labels_dir)[[target, "cohort"]]
    metadata.index.name = "sample"
    metadata = metadata.rename(columns={target: "target"})
    for concept_graphs_dir in concept_graphs_dirs:
        concept_name = concept_graphs_dir.name
        for graph_path in concept_graphs_dir.glob("*.pt"):
            sample = graph_path.stem
            g = torch.load(graph_path)
            metadata.loc[sample, f"{concept_name}__num_nodes"] = g.num_nodes
            metadata.loc[sample, f"{concept_name}__graph_path"] = graph_path

    # assert metadata.isna().any() == False
    return metadata


def filter_samples(metadata, min_num_nodes: int) -> pd.DataFrame:
    targets = metadata["target"]
    # TODO: how to handle nan strings?
    valid_samples = metadata[(targets.notna()) & (targets != "nan")]

    cols = valid_samples.columns.str.endswith("num_nodes")
    m = (valid_samples.loc[:, cols] >= min_num_nodes).all(1)
    valid_samples = valid_samples[m]

    assert valid_samples.isna().any().any() == False
    return valid_samples


def filter_mask_objects(mask: np.ndarray, labels: pd.Series, include_labels: list):
    m = labels.isin(include_labels)
    obj_ids = set(labels[m].index.get_level_values("cell_id"))
    mask_objs = set(mask.flatten())
    mask_objs.remove(0)

    for obj in mask_objs - obj_ids:
        mask[mask == obj] = 0

    return mask
