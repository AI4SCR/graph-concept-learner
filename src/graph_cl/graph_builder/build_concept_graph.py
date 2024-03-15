import logging
from pathlib import Path

import torch
from ..data_models.Concept import ConceptConfig
from ..preprocessing.filter import filter_mask_objects
from ..graph_builder.build_graph import build_pyg_graph
from ..data_models.Sample import Sample


def build_concept_graph(sample: Sample, concept_config: ConceptConfig):
    labels = sample.labels.cell_class
    mask = sample.mask
    mask = filter_mask_objects(
        mask, labels=labels, include_labels=concept_config.filter.include_labels
    )

    graph = build_pyg_graph(
        mask=mask,
        topology=concept_config.graph.topology,
        params=concept_config.graph.params,
    )

    # TODO: should we just add the whole sample to the graph?
    # graph.sample = sample
    graph.sample_id = sample.id
    graph.sample_name = sample.name
    graph.cohort = sample.cohort
    graph.concept_name = concept_config.name

    return graph


def build_concept_graph_from_paths(
    sample_path: Path,
    concept_config_path: Path,
    concept_graph_path: Path,
    overwrite: bool = False,
):
    concept_config = ConceptConfig.from_yaml(concept_config_path)
    sample = Sample.model_validate_from_file(sample_path)

    if concept_graph_path.exists() and overwrite is False:
        logging.info(
            f"Concept graph for {concept_config.name} and {sample.name} exists.\n...skipping"
        )
        return
    elif concept_graph_path.exists() and overwrite is True:
        logging.info(
            f"Concept graph for {concept_config.name} and {sample.name} exists.\n...overwriting"
        )

    graph = build_concept_graph(sample, concept_config)

    graph.sample_id = sample.id
    graph.sample_name = sample.name
    graph.concept_name = concept_config.name

    # note: this will enable us to transition to a Sample() object that can load lazily
    concept_graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, concept_graph_path)
