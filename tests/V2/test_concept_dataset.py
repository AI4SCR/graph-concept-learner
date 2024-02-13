from graph_cl.datasets.ConceptDatasetV2 import ConceptDataset
from pathlib import Path


def test_concept_dataset():
    experiment_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2"
    )
    data_dir = Path("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data")
    ConceptDataset(experiment_dir, data_dir)
