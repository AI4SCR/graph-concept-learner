import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def split_basel_leave_zurich_as_external(
    samples: pd.DataFrame, n_folds: int, train_size: float
) -> list[pd.DataFrame]:
    samples = samples.assign(split=None)

    bs = samples[samples.cohort == "basel"]
    zh = samples[samples.cohort == "zurich"]
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
    raise NotImplementedError()


def split_zurich_leave_basel_as_external():
    raise NotImplementedError()


SPLIT_STRATEGIES = {
    "split_basel_leave_zurich_as_external": split_basel_leave_zurich_as_external,
}


def split_samples_into_folds(
    samples, split_strategy: str, **kwargs
) -> list[pd.DataFrame]:
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    func = SPLIT_STRATEGIES[split_strategy]
    folds = func(samples, **kwargs)
    return folds
