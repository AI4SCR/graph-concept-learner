import shutil

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
from ...preprocessing.attribute import prepare_attributes
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
    # TODO: this filtering is done twice atm, also in `preprocess_samples`
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

    # delete previous samples, attributes and datasets to avoid leakage
    shutil.rmtree(experiment.samples_dir, ignore_errors=True)
    shutil.rmtree(experiment.attributes_dir, ignore_errors=True)
    shutil.rmtree(experiment.datasets_dir, ignore_errors=True)
    experiment.split_info_path.unlink(missing_ok=True)
    experiment.create_folder_hierarchy()
    # TODO: do we need to delete the predictions and results as well?

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

    concept_names = data_config.concepts
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
    for stage, samples in splits.items():
        for sample in samples:
            sample_path = experiment.get_sample_path(
                sample_name=sample.name, stage=stage
            )
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample.model_dump_to_json(sample_path)
    split_info.to_parquet(experiment.split_info_path)

    # precomputed_attrs = all(
    #     map(lambda x: x in experiment.get_attribute_path(x.stage, x.sample_name).exists(), split_info))
    logger.info(f"Computing attributes for samples.")
    if experiment.attributes_dir.exists():
        shutil.rmtree(experiment.attributes_dir)
    prepare_attributes(splits, experiment, data_config)


def load_splits(experiment: Experiment) -> dict[str, list[Sample]]:
    splits = {}
    for stage in ["fit", "val", "test", "predict"]:
        split_dir = experiment.samples_dir / stage
        if split_dir.exists():
            splits[stage] = [
                Sample.model_validate_from_json(i) for i in split_dir.glob("*.json")
            ]
    return splits


# TODO: we could introduce a `model_name` to pretrain different GNN models, but probably it would be better to do that
#  in a separate experiment
def pretrain(experiment_name: str, concept_name: str):
    experiment = Experiment(project=project, name=experiment_name)
    data_config = DataConfig.model_validate_from_json(experiment.data_config_path)

    assert isinstance(concept_name, str)
    assert concept_name in data_config.concepts

    splits = load_splits(experiment=experiment)

    dm = ConceptDataModule(
        splits=splits,
        concepts=concept_name,
        datasets_dir=experiment.datasets_dir / "gnn" / concept_name,
    )

    model_config = ModelGNNConfig.model_validate_from_yaml(
        experiment.model_gnn_config_path
    )
    model_config.MLP.out_channels = dm.num_classes
    model_config.GNN.kwargs.in_channels = dm.num_features

    train_config = TrainConfig.model_validate_from_json(experiment.pretrain_config_path)
    train_config.tracking.checkpoint_dir = experiment.get_concept_model_dir(
        model_name="gnn", concept_name=concept_name
    )
    train_config.tracking.checkpoint_dir = None

    module = LitGNN(model_config=model_config.dict(), train_config=train_config)
    train_model(module, dm, train_config)


def train(experiment_name: str):
    experiment = Experiment(project=project, name=experiment_name)
    data_config = DataConfig.model_validate_from_json(experiment.data_config_path)

    splits = load_splits(experiment=experiment)

    dm = ConceptDataModule(
        splits=splits,
        concepts=data_config.concepts,
        datasets_dir=experiment.datasets_dir / "gcl",
    )

    model_config = ModelGCLConfig.model_validate_from_yaml(
        experiment.model_gcl_config_path
    )
    model_config.num_classes = dm.num_classes
    model_config.in_channels = dm.num_features

    concept_graph_ckpts = {
        concept_name: experiment.get_concept_model_dir("gnn", concept_name)
        / "best_model.ckpt"
        for concept_name in data_config.concepts
    }

    train_config = TrainConfig.model_validate_from_json(experiment.train_config_path)
    train_config.tracking.checkpoint_dir = experiment.get_model_dir("gcl")
    train_config.tracking.predictions_dir = None

    gcl = LitGCL(
        concept_graph_ckpts=concept_graph_ckpts,
        model_config=model_config.dict(),
        train_config=train_config,
    )

    train_model(gcl, dm, train_config)


def test(experiment_name: str):
    experiment = Experiment(project=project, name=experiment_name)
    data_config = DataConfig.model_validate_from_json(experiment.data_config_path)

    splits = load_splits(experiment=experiment)

    (experiment.datasets_dir / "gcl").mkdir(parents=True, exist_ok=True)
    dm = ConceptDataModule(
        splits=splits,
        concepts=data_config.concepts,
        datasets_dir=experiment.datasets_dir / "gcl",
    )

    gcl = LitGCL.load_from_checkpoint(
        experiment.get_model_dir("gcl") / "best_model.ckpt"
    )
    gcl.config.tracking.predictions_dir = experiment.predictions_dir / "gcl"
    (experiment.predictions_dir / "gcl").mkdir(parents=True, exist_ok=True)

    import lightning as L

    trainer = L.Trainer(**gcl.config.trainer.dict())
    trainer.test(model=gcl, datamodule=dm)
