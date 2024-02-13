import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


def split_basel_leave_zurich_as_external(
    valid_samples: pd.DataFrame, n_folds: int, train_size: float
) -> list[pd.DataFrame]:

    valid_samples = valid_samples.assign(split=None)

    bs = valid_samples[valid_samples.cohort == "basel"]
    zh = valid_samples[valid_samples.cohort == "zurich"]
    zh = zh.assign(split="test")

    sss = StratifiedShuffleSplit(
        n_splits=n_folds, train_size=train_size, random_state=0
    )
    split_iter = sss.split(bs.index.values, bs["target"])

    folds = []
    for i, idcs in enumerate(split_iter):
        train_idc, val_idc = idcs
        bs.iloc[train_idc, bs.columns.get_loc("split")] = "train"
        bs.iloc[val_idc, zh.columns.get_loc("split")] = "val"

        fold = pd.concat([bs, zh])
        folds.append(fold)
    return folds


def split_both_cohorts():
    pass


def split_zurich_leave_basel_as_external():
    pass


def create_folds(
    method: str,
    valid_samples_path: Path,
    n_folds: float,
    train_size: float,
    output_dir: Path,
):

    output_dir.mkdir(parents=True, exist_ok=True)
    valid_samples = pd.read_csv(valid_samples_path, index_col="core")

    if method == "split_basel_leave_zurich_as_external":
        split_basel_leave_zurich_as_external(
            valid_samples, n_folds, train_size, output_dir
        )
    elif method == "split_both_cohorts":
        split_both_cohorts()
    elif method == "split_zurich_leave_basel_as_external":
        split_zurich_leave_basel_as_external()
    else:
        raise ValueError(f"Unknown split method: {method}")


SPLIT_STRATEGIES = {
    "split_basel_leave_zurich_as_external": split_basel_leave_zurich_as_external,
}
