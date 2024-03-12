import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..preprocessing.split import SPLIT_STRATEGIES
from ..preprocessing.filter import collect_metadata, filter_samples
from ..data_models.Sample import Sample


def prepare_samples(samples: list[Sample], data_config):
    metadata = collect_metadata(
        samples=samples,
        target=data_config.target,
    )
    samples = filter_samples(metadata=metadata, **data_config.filter.dict())

    # TODO: it would be better to encode on the train split only
    target_encoder = LabelEncoder()
    targets_encoded = target_encoder.fit_transform(samples.target)
    samples = samples.assign(y=targets_encoded)
    assert samples.isna().any().any() == False

    func = SPLIT_STRATEGIES[data_config.split.strategy]
    # TODO: remove
    import numpy as np

    samples = samples.assign(
        target=np.random.choice(["negative", "positive"], len(samples))
    )

    samples = func(samples, **data_config.split.kwargs)


def prepare_samples_from_files(
    processed_dir,
):
    samples_pp = pd.read_parquet(processed_dir / "samples.parquet")
