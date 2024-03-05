from graph_cl.dataloader.ConceptDataset import CptDatasetMemo
from pathlib import Path
import pandas as pd


def test_in_memory_dataset():
    fold_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
    )
    concept = "knn_all"
    ds = CptDatasetMemo(fold_path, concept)
    fold_info = pd.read_parquet(fold_path / "info.parquet")

    ds_train = CptDatasetMemo(fold_path, concept)
    ds_val = CptDatasetMemo(fold_path, concept, split="val")
    ds_test = CptDatasetMemo(fold_path, concept, split="test")

    split_counts = fold_info["split"].value_counts()

    assert len(ds) == split_counts.loc["train"]
    assert len(ds_train) == split_counts.loc["train"]
    assert len(ds_test) == split_counts.loc["test"]
    assert len(ds_val) == split_counts.loc["val"]
