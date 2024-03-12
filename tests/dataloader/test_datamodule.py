from graph_cl.dataloader.ConceptDataModule import ConceptDataModule
from graph_cl.data_models.Data import DataConfig
from pathlib import Path
import yaml


def test_ConceptDataModule():
    labels_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed/labels/samples/"
    )
    concept_graphs_dirs = [
        p
        for p in Path(
            "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs"
        ).iterdir()
        if p.is_dir()
    ]
    processed_dir: Path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
    )
    config_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/data.yaml"
    )
    batch_size = 8
    shuffle = True
    cache_dir = None

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = DataConfig(**config)

    concept_data_module = ConceptDataModule(
        labels_dir=labels_dir,
        concept_graphs_dirs=concept_graphs_dirs,
        processed_dir=processed_dir,
        config=config,
        batch_size=batch_size,
        shuffle=shuffle,
        cache_dir=cache_dir,
    )
    concept_data_module.setup(stage="fit")
