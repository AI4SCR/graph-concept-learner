from ...datasets import get_dataset_by_name
from ...data_models.Project import project
from ...data_models.Experiment import GCLExperiment as Experiment
from ...data_models.Data import DataConfig
from ...data_models.Sample import Sample
from ...data_models.Concept import ConceptConfig
import torch
from ai4bmr_core.log.log import logger


def create_concept_graph(experiment_name: str, sample_name: str, concept_name: str):
    from graph_cl.graph_builder.build_concept_graph import build_concept_graph

    experiment = Experiment(project=project, name=experiment_name)
    data_config = DataConfig.model_validate_from_json(experiment.data_config_path)

    ds = get_dataset_by_name(dataset_name=data_config.dataset_name)

    sample_path = ds.get_sample_path(sample_name)
    sample = Sample.model_validate_from_json(sample_path)

    if sample.labels_url is None or sample.mask_url is None:
        # TODO: should we raise an exception here?
        logger.warn(
            f"{sample_name} ignored because it does not have labels 🏷️ or mask 🎭."
        )

    concept_path = project.concepts_dir / f"{concept_name}.yaml"
    concept_config = ConceptConfig.model_validate_from_json(concept_path)

    graph = build_concept_graph(sample=sample, concept_config=concept_config)
    concept_graph_path = ds.get_concept_graph_path(concept_name, sample_name)
    concept_graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, concept_graph_path)

    sample.concept_graph_url[concept_name] = concept_graph_path
    sample.model_dump_to_json(ds.samples_dir / f"{sample_name}.json")
