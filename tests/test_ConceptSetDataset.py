from graph_cl.dataloader.ConceptSetDataset import ConceptSetDataset
from pathlib import Path
import pandas as pd
from torch_geometric.loader import DataLoader


def test_in_memory_dataset():
    fold_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
    )
    fold_info = pd.read_parquet(fold_path / "info.parquet")

    ds_train = ConceptSetDataset(root=fold_path, fold_info=fold_info, split="train")
    ds_val = ConceptSetDataset(root=fold_path, fold_info=fold_info, split="val")
    ds_test = ConceptSetDataset(root=fold_path, fold_info=fold_info, split="test")

    s1 = ds_train[0]
    s2 = ds_train[1]

    dl = DataLoader(ds_train, batch_size=2, shuffle=False)

    batch = next(iter(dl))
    c1 = batch["radius_tumor_immune"]
    c2 = batch["knn_all"]

    split_counts = fold_info["split"].value_counts()

    assert len(ds_train) == split_counts.loc["train"]
    assert len(ds_test) == split_counts.loc["test"]
    assert len(ds_val) == split_counts.loc["val"]
