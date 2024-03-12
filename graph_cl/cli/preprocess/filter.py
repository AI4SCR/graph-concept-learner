from ...data_models.Data import DataConfig
from ...data_models.Sample import Sample

from pathlib import Path
import pandas as pd


def filter(samples: list[Sample], data_config: DataConfig):
    samples = list(filter(lambda x: pd.notna(x.target), samples))
    for concept_graph_name in data_config.concepts:
        samples = filter(
            lambda x: x.concept_graphs[concept_graph_name].num_nodes
            > data_config.filter.min_num_nodes,
            samples,
        )
    return samples


def filter_from_paths(data_config_path: Path):
    data_config = DataConfig.from_yaml(data_config_path)
