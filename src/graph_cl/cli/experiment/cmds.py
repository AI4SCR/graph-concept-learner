import pandas as pd
import torch
from ai4bmr_core.log.log import logger, enable_verbose

from ...data_models.Concept import ConceptConfig
from ...data_models.Data import DataConfig
from ...data_models.Experiment import GCLExperiment as Experiment
from ...data_models.Model import ModelGCLConfig
from ...data_models.Model import ModelGNNConfig
from ...data_models.Project import project
from ...data_models.Sample import Sample
from ...data_models.Train import TrainConfig
from ...dataloader.ConceptDataModule import ConceptDataModule
from ...datasets import get_dataset_by_name
from ...preprocessing.encode import encode_target
from ...preprocessing.filter import (
    filter_has_mask,
    filter_has_labels,
    filter_has_target,
    filter_has_enough_nodes,
)
from ...preprocessing.split import split_samples
from ...train.lightning import LitGCL
from ...train.lightning import LitGNN
from ...train.train import train_model


def create_concept_graph(
    experiment_name: str, sample_name: str, concept_name: str, force: bool = False
):
    from graph_cl.graph_builder.build_concept_graph import build_concept_graph

    experiment = Experiment(project=project, name=experiment_name)
    data_config = DataConfig.model_validate_from_json(experiment.data_config_path)

    ds = get_dataset_by_name(dataset_name=data_config.dataset_name)()
    concept_graph_path = ds.get_concept_graph_path(concept_name, sample_name)
    concept_graph_path.parent.mkdir(parents=True, exist_ok=True)

    if concept_graph_path.exists() and not force:
        logger.info(
            # f"Concept graph for {sample_name} and concept {concept_name} already exists at {concept_graph_path}. Skipping.")
            f"Concept graph for {sample_name} and concept {concept_name} already exists. ⏭️ Skipping..."
        )
        return
    elif force:
        logger.info(
            f"Recompute concept graph for {sample_name} and concept {concept_name}"
        )

    sample_path = ds.get_sample_path_by_name(sample_name)
    sample = Sample.model_validate_from_json(sample_path)
    if sample.labels_url is None or sample.mask_url is None:
        # TODO: should we raise an exception here?
        logger.warn(
            f"{sample_name} ignored because it does not have labels 🏷️ or a mask 🎭."
        )
        return

    concept_path = project.get_concept_config_path(concept_name)
    concept_config = ConceptConfig.model_validate_from_json(concept_path)

    graph = build_concept_graph(sample=sample, concept_config=concept_config)
    torch.save(graph, concept_graph_path)

    sample.concept_graph_url[concept_name] = concept_graph_path
    sample.model_dump_to_json(ds.samples_dir / f"{sample_name}.json")


@enable_verbose
def preprocess_samples(experiment_name: str):
    experiment = Experiment(project=project, name=experiment_name)

    data_config = DataConfig.model_validate_from_json(experiment.data_config_path)
    ds = get_dataset_by_name(dataset_name=data_config.dataset_name)()
    samples = ds._data
    n_samples = len(samples)

    logger.info(f"Preprocessing dataset `{ds._name}` with {len(samples)}.")

    samples = filter_has_mask(samples)
    logger.info(
        f"Filtered out {n_samples - len(samples)} with no mask 🎭. Run with -vvv to see which ones."
    )
    n_samples = len(samples)

    samples = filter_has_labels(samples)
    logger.info(
        f"Filtered out {n_samples - len(samples)} with no labels 🏷️. Run with -vvv to see which ones."
    )
    n_samples = len(samples)

    logger.info(f"Setting target `{data_config.target}` 🎯 on samples.")
    target = data_config.target
    for sample in samples:
        sample.target = sample.sample_labels[target]
    samples = encode_target(samples)

    samples = filter_has_target(samples)
    logger.info(
        f"Filtered out {n_samples - len(samples)} with no target 🎯. Run with -vvv to see which ones."
    )
    n_samples = len(samples)

    # NOTE: this is a bit of a hack, but we need to load the model config to get the concept names
    #   this is why we load the model config with dummy values for `num_classes` and `in_channels`
    #   ‼️I am convinced that the concepts belong to the model config not the data config
    #   🤔An alternative could be to have negative default values for these fields to trigger an error at model
    #   instantiation.
    model_config = ModelGCLConfig.model_validate_from_json(
        experiment.model_gcl_config_path, num_classes=-1, in_channels=-1
    )
    concept_names = model_config.concepts
    min_num_nodes = data_config.filter.min_num_nodes
    samples = filter_has_enough_nodes(
        samples, concept_names, min_num_nodes=min_num_nodes
    )
    logger.info(
        f"Filtered out {n_samples - len(samples)} with not enough nodes 🕸️. Run with -vvv to see which ones."
    )

    n_samples = len(samples)
    logger.info(f"Splitting {n_samples} samples in train, validation and test splits.")

    splits, split_info = split_samples(samples, data_config)
    split_info = split_info.assign(
        sample_url=split_info.sample_name.map(
            lambda x: str(ds.samples_dir / f"{x}.json")
        )
    )
    split_info.to_parquet(experiment.split_info_path)
    for stage, samples in splits.items():
        for sample in samples:
            sample_path = experiment.get_sample_path(
                sample_name=sample.name, stage=stage
            )
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample.model_dump_to_json(sample_path)


def load_splits(
    split_info: pd.DataFrame, experiment: Experiment
) -> dict[str, list[Sample]]:
    splits = {
        # note: here we load from the dataset samples, i.e. with undefined split attribute
        # stage: [Sample.model_validate_from_file(x.sample_url) for _, x in split_info.loc[stage].iterrows()]
        # note: here we load from the experiment samples, created when dataset is split for the experiment
        stage: [
            Sample.model_validate_from_json(experiment.get_sample_path(x.sample_name))
            for _, x in split_info.loc[stage].iterrows()
        ]
        for stage in split_info.index.unique()
    }

    return splits


def pretrain(experiment_name: str, concept_name: str):
    experiment = Experiment(experiment_name=experiment_name)
    data_config = DataConfig.model_validate_from_json(experiment.data_config_path)

    split_info = pd.read_parquet(experiment.split_info_path).set_index("stage")
    splits = load_splits(split_info, experiment)

    num_classes = split_info.target.nunique()
    num_features = splits["fit"][0].expression.shape[1]
    model_config = ModelGNNConfig.model_validate_from_json(
        experiment.model_gnn_config_path,
        num_classes=num_classes,
        in_channels=num_features,
    )
    assert (
        concept_name
        in ModelGCLConfig.model_validate_from_json(
            experiment.model_gcl_config_path, num_classes=-1, in_channels=-1
        ).concepts
    )

    dm = ConceptDataModule(
        splits=splits,
        model_name=model_config.name,
        concepts=concept_name,
        config=data_config,
        factory=experiment,
        force_attr_computation=True,
    )

    train_config = TrainConfig.model_validate_from_json(experiment.pretrain_config_path)
    train_config.tracking.checkpoint_dir = experiment.get_concept_model_dir(
        concept_name
    )

    module = LitGNN(model_config=model_config, train_config=train_config)
    train_model(module, dm, train_config)


def train(experiment_name: str):
    factory = Experiment(experiment_name=experiment_name)
    data_config = DataConfig.model_validate_from_json(factory.data_config_path)

    split_info = pd.read_parquet(factory.split_info_path).set_index("stage")
    splits = load_splits(split_info, factory)

    num_features = splits["fit"][0].expression.shape[1]
    num_classes = split_info.target.nunique()
    model_config = ModelGCLConfig.model_validate_from_json(
        factory.model_gcl_config_path, num_classes=num_classes, in_channels=num_features
    )

    dm = ConceptDataModule(
        splits=splits,
        model_name=model_config.name,
        concepts=model_config.concepts,
        config=data_config,
        factory=factory,
    )

    concept_graph_ckpts = {
        concept_name: factory.get_concept_model_dir(concept_name) / "best_model.ckpt"
        for concept_name in model_config.concepts
    }

    train_config = TrainConfig.model_validate_from_json(factory.train_config_path)
    train_config.tracking.checkpoint_dir = factory.get_model_dir("gcl")

    gcl = LitGCL(
        concept_graph_ckpts=concept_graph_ckpts,
        model_config=model_config,
        train_config=train_config,
    )

    train_model(gcl, dm, train_config)
