from graph_cl.datasets.ConceptDataset import ConceptDataset, CptDatasetMemo
from graph_cl import CONFIG
import pandas as pd

root = CONFIG.project.root / "experiments" / CONFIG.experiment.name / "data"
concept = "knn"
fold_meta_data = pd.read_csv(root / "folds" / "fold_0.csv")

# concept_dataset = ConceptDataset(root / "graphs"/ concept, fold_meta_data)
# concept_dataset[0]

concept_dataset = CptDatasetMemo(root / "graphs" / concept, fold_meta_data)
