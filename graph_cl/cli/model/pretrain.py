from pathlib import Path
import click
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from ...data_models.Model import ModelGNNConfig
from ...data_models.Train import TrainConfig
from ...dataloader.ConceptDataModule import ConceptDataModule
from ...preprocessing.split import SPLIT_STRATEGIES
from ...data_models.Data import DataConfig
from ...models.gnn import GNN_plus_MPL
from ...train.lightning import LitGNN
from ...train.train import train
from ...preprocessing.filter import collect_metadata, filter_samples
from ...preprocessing.attribute import collect_features


@click.command()
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("concept_name", type=str)
def pretrain(data_dir: Path, experiment_dir: Path, concept_name: str):
    data_config_path = experiment_dir / "configuration" / "data.yaml"
    model_config_path = experiment_dir / "configuration" / "model_gnn.yaml"
    train_config_path = experiment_dir / "configuration" / "pretrain.yaml"

    processed_dir = data_dir / "02_processed"
    labels_dir = processed_dir / "labels" / "samples"

    concept_graphs_dir = data_dir / "03_concept_graphs"

    with open(data_config_path, "r") as f:
        data_config = yaml.load(f, Loader=yaml.Loader)
        data_config = DataConfig(**data_config)

    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.Loader)
        model_config = ModelGNNConfig(**model_config)

    with open(train_config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.Loader)
        train_config = TrainConfig(**train_config)

    samples_pp = pd.read_parquet(processed_dir / "samples.parquet")
    samples_info = collect_metadata(
        target=data_config.target,
        labels_dir=labels_dir,
        concept_graphs_dirs=[p for p in concept_graphs_dir.iterdir() if p.is_dir()],
    )
    assert set(samples_pp.index) == set(samples_info.index)
    samples = pd.concat([samples_pp, samples_info], axis=1)
    samples = filter_samples(metadata=samples, **data_config.filter.dict())

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
    feats = collect_features(samples=samples, config=data_config.features)

    datamodule = ConceptDataModule(
        samples=samples, features=feats, config=data_config, concepts=[concept_name]
    )

    model_config.num_classes = len(set(samples.y))
    model_config.in_channels = feats.shape[1]

    model = GNN_plus_MPL(model_config.dict())
    module = LitGNN(model, config=train_config)

    train(module, datamodule, train_config)
