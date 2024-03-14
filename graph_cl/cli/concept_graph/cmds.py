from ...data_models.DatasetPathFactory import DatasetPathFactory
from ...data_models.ProjectSettings import ProjectSettings
from ...data_models.Sample import Sample
from ...data_models.Concept import ConceptConfig
import torch


def create_concept_graph(sample_name: str, dataset_name: str, concept_name: str):
    from graph_cl.graph_builder.build_concept_graph import build_concept_graph

    ps = ProjectSettings(dataset_name=dataset_name)
    ds = DatasetPathFactory(dataset_name=dataset_name)

    if (ds.samples_dir / f"{sample_name}.json").exists():
        # we load the samples from 04_samples if it exists
        sample_path = ds.samples_dir / f"{sample_name}.json"
    else:
        # we load the sample form 02_processed/samples
        sample_path = ds.processed_samples_dir / f"{sample_name}.json"
    sample = Sample.model_validate_from_file(sample_path)

    if sample.labels_url is None or sample.mask_url is None:
        raise ValueError(f"Sample {sample_name} does not have labels or mask")

    concept_path = ps.concepts_dir / f"{concept_name}.yaml"
    concept_config = ConceptConfig.from_yaml(concept_path)

    graph = build_concept_graph(sample=sample, concept_config=concept_config)
    concept_graph_path = ds.get_concept_graph_path(concept_name, sample_name)
    concept_graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, concept_graph_path)

    ds.samples_dir.mkdir(parents=True, exist_ok=True)
    sample.concept_graph_url[concept_name] = concept_graph_path
    sample.model_dump_to_file(ds.samples_dir / f"{sample_name}.json")
