from ...preprocessing.split import SPLIT_STRATEGIES
from ...data_models.Sample import Sample
from ...data_models.Data import DataConfig

import pandas as pd


def split_samples(samples: list[Sample], config: DataConfig):
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
    return pd.DataFrame(cont)
