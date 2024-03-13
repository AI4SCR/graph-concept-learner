from ...data_models.Data import DataConfig
from ...data_models.Sample import Sample

import pandas as pd


def filter_samples(samples: list[Sample], data_config: DataConfig):
    samples = filter(lambda x: pd.notna(x.target), samples)
    for concept_graph_name in data_config.concepts:
        samples = filter(
            lambda x: x.concept_graphs[concept_graph_name].num_nodes
            > data_config.filter.min_num_nodes,
            samples,
        )
    return list(samples)
