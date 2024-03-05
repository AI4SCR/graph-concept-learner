import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split


def train_val_basel_test_zurich(
    samples: pd.DataFrame,
    train_size: float = None,
    test_size: float = None,
    random_state: int = None,
) -> pd.DataFrame:
    samples = samples.assign(stage=None)

    bs = samples[samples.cohort == "basel"]
    test = samples[samples.cohort == "zurich"]
    test = test.assign(stage="test")

    train, val = train_test_split(
        bs,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        stratify=bs["target"],
    )

    # train = pd.DataFrame(train, columns=bs.columns, index=bs.index)
    # val = pd.DataFrame(val, columns=bs.columns, index=bs.index)

    train = train.assign(stage="fit")
    val = val.assign(stage="val")

    return pd.concat([train, val, test])


def split_both_cohorts():
    raise NotImplementedError()


def split_zurich_leave_basel_as_external():
    raise NotImplementedError()


SPLIT_STRATEGIES = {
    "split_basel_leave_zurich_as_external": train_val_basel_test_zurich,
}


# def split_samples_into_folds(
#         samples, split_strategy: str, **kwargs
# ) -> list[pd.DataFrame]:
#     if split_strategy not in SPLIT_STRATEGIES:
#         raise ValueError(f"Unknown split strategy: {split_strategy}")
#
#     func = SPLIT_STRATEGIES[split_strategy]
#     folds = func(samples, **kwargs)
#     return folds
