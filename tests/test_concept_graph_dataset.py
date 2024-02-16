from graph_cl.datasets.ConceptDatasetV2 import ConceptDataset
from pathlib import Path
import pandas as pd

experiment_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2"
)
data_dir = Path("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data")
ds = ConceptDataset(experiment_path, data_dir)
