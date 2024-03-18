from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from ..data_models.Data import DataConfig
from ..data_models.Sample import Sample


def encode_target(samples: list[Sample]) -> list[Sample]:
    target_encoder = LabelEncoder()
    targets = [s.target for s in samples]
    target_encoder.fit(targets)
    for s in samples:
        s.target_encoded = target_encoder.transform([s.target])[0]
    return samples


def encode_target_from_paths(samples_dir: Path, data_config_path: Path):
    samples = [Sample.from_pickle(p) for p in samples_dir.glob("*.pkl")]
    data_config = DataConfig.model_validate_from_json(data_config_path)
    samples = encode_target(samples, data_config)
    return samples
