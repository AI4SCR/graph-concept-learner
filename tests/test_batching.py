from torch_geometric.loader import DataLoader
from graph_cl.datasets.ConceptDataset import CptDatasetMemo
from pathlib import Path
import pandas as pd

# %%

fold_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
)

fold_info = pd.read_parquet(fold_dir / "info.parquet")

# %%
# Load dataset
concept = "knn_all"
ds_train = CptDatasetMemo(
    root=fold_dir, fold_info=fold_info, concept=concept, split="train"
)

dl_train = DataLoader(
    ds_train,
    batch_size=3,
    shuffle=False,
)

batch = next(iter(dl_train))
