from sklearn.model_selection import train_test_split
from ..data_models.Sample import Sample
from ..data_models.Data import DataConfig

import pandas as pd


def train_val_basel_test_zurich(
    samples: list[Sample],
    train_size: float = None,
    test_size: float = None,
    random_state: int = None,
) -> dict[str, list[Sample]]:
    test = list(filter(lambda s: s.cohort == "zurich", samples))
    for s in test:
        s.split = "test"

    bs = list(filter(lambda s: s.cohort == "basel", samples))
    stratify = [s.target for s in samples]
    train, val = train_test_split(
        bs,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

    for s in train:
        s.split = "fit"

    for s in val:
        s.split = "val"

    return {"fit": train, "val": val, "test": test}


def split_both_cohorts():
    raise NotImplementedError()


def split_zurich_leave_basel_as_external():
    raise NotImplementedError()


SPLIT_STRATEGIES = {
    "split_basel_leave_zurich_as_external": train_val_basel_test_zurich,
}


def split_samples(
    samples: list[Sample], config: DataConfig
) -> (dict[str, Sample], pd.DataFrame):
    func = SPLIT_STRATEGIES[config.split.strategy]
    splits = func(samples, **config.split.kwargs)

    cont = []
    for stage in ["fit", "val", "test"]:
        cont.extend(
            [
                {
                    "sample_id": s.id,
                    "sample_name": s.name,
                    "target": s.target,
                    "split": stage,
                    "stage": stage,
                }
                for s in splits[stage]
            ]
        )
    return splits, pd.DataFrame(cont)
