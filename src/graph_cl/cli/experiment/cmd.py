import pandas as pd

from ...data_models.Data import DataConfig
from ...data_models.Sample import Sample
from ...data_models.ExperimentPathFactory import ExperimentPathFactory
from ...data_models.DatasetPathFactory import DatasetPathFactory
from ...preprocessing.filter import filter_samples
from ...preprocessing.encode import encode_target
from ...preprocessing.split import split_samples
from ...data_models.Model import ModelGNNConfig
from ...train.lightning import LitGNN
from ...data_models.Train import TrainConfig
from ...train.train import train_model
from ...dataloader.ConceptDataModule import ConceptDataModule
from ...data_models.Model import ModelGCLConfig
from ...train.lightning import LitGCL
from ...utils.log import logger


def preprocess_samples(experiment_name: str):
    factory = ExperimentPathFactory(experiment_name=experiment_name)
    factory.init()

    data_config = DataConfig.from_yaml(factory.data_config_path)
    ds = DatasetPathFactory(dataset_name=data_config.dataset_name)

    samples = [
        Sample.model_validate_from_file(p) for p in ds.samples_dir.glob("*.json")
    ]

    samples = encode_target(samples, data_config)

    # NOTE: this is a bit of a hack, but we need to load the model config to get the concept names
    #   this is why we load the model config with dummy values for `num_classes` and `in_channels`
    #   ‼️I am convinced that the concepts belong to the model config not the data config
    #   🤔An alternative could be to have negative default values for these fields to trigger an error at model
    #   instantiation.
    model_config = ModelGCLConfig.from_yaml(
        factory.model_gcl_config_path, num_classes=-1, in_channels=-1
    )
    samples = filter_samples(
        samples,
        concept_names=model_config.concepts,
        min_num_nodes=data_config.filter.min_num_nodes,
    )

    splits, split_info = split_samples(samples, data_config)
    split_info = split_info.assign(
        sample_url=split_info.sample_name.map(
            lambda x: str(ds.samples_dir / f"{x}.json")
        )
    )
    split_info.to_parquet(factory.split_info_path)
    for samples in splits.values():
        for sample in samples:
            sample.model_dump_to_file(factory.get_sample_path(sample.name))


def load_splits(
    split_info: pd.DataFrame, factory: ExperimentPathFactory
) -> dict[str, list[Sample]]:
    splits = {
        # note: here we load from the dataset samples, i.e. with undefined split attribute
        # stage: [Sample.model_validate_from_file(x.sample_url) for _, x in split_info.loc[stage].iterrows()]
        # note: here we load from the experiment samples, created when dataset is split for the experiment
        stage: [
            Sample.model_validate_from_file(factory.get_sample_path(x.sample_name))
            for _, x in split_info.loc[stage].iterrows()
        ]
        for stage in split_info.index.unique()
    }

    return splits


def pretrain(experiment_name: str, concept_name: str):
    factory = ExperimentPathFactory(experiment_name=experiment_name)
    data_config = DataConfig.from_yaml(factory.data_config_path)

    split_info = pd.read_parquet(factory.split_info_path).set_index("stage")
    splits = load_splits(split_info, factory)

    num_classes = split_info.target.nunique()
    num_features = splits["fit"][0].expression.shape[1]
    model_config = ModelGNNConfig.from_yaml(
        factory.model_gnn_config_path, num_classes=num_classes, in_channels=num_features
    )
    assert (
        concept_name
        in ModelGCLConfig.from_yaml(
            factory.model_gcl_config_path, num_classes=-1, in_channels=-1
        ).concepts
    )

    dm = ConceptDataModule(
        splits=splits,
        model_name=model_config.name,
        concepts=concept_name,
        config=data_config,
        factory=factory,
        force_attr_computation=True,
    )

    train_config = TrainConfig.from_yaml(factory.pretrain_config_path)
    train_config.tracking.checkpoint_dir = factory.get_concept_model_dir(concept_name)

    module = LitGNN(model_config=model_config, train_config=train_config)
    train_model(module, dm, train_config)


def train(experiment_name: str):
    factory = ExperimentPathFactory(experiment_name=experiment_name)
    data_config = DataConfig.from_yaml(factory.data_config_path)

    split_info = pd.read_parquet(factory.split_info_path).set_index("stage")
    splits = load_splits(split_info, factory)

    num_features = splits["fit"][0].expression.shape[1]
    num_classes = split_info.target.nunique()
    model_config = ModelGCLConfig.from_yaml(
        factory.model_gcl_config_path, num_classes=num_classes, in_channels=num_features
    )

    dm = ConceptDataModule(
        splits=splits,
        model_name=model_config.name,
        concepts=model_config.concepts,
        config=data_config,
        factory=factory,
    )

    train_config = TrainConfig.from_yaml(factory.train_config_path)
    train_config.tracking.checkpoint_dir = factory.model_gcl_dir

    concept_graph_ckpts = {
        concept_name: factory.get_concept_model_dir(concept_name) / "best_model.ckpt"
        for concept_name in model_config.concepts
    }

    gcl = LitGCL(
        concept_graph_ckpts=concept_graph_ckpts,
        model_config=model_config,
        train_config=train_config,
    )

    train_model(gcl, dm, train_config)
