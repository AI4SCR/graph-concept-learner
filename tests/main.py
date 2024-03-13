import pandas as pd
import torch

from graph_cl.data_models.ProjectSettings import ProjectSettings
from graph_cl.data_models.Sample import Sample
from graph_cl.datasets.Jackson import Jackson

from graph_cl.cli.create.create import (
    project as create_project,
    dataset as create_dataset,
    experiment as create_experiment,
)

##### END OF 1 #####
create_project()

##### END OF 1 #####
dataset_name = Jackson.name
create_dataset(dataset_name)
# now copy all the data to the raw_dir
# for i in $(find ../../../archive/data_old/01_raw/ -name "*.zip"); do echo ln -s $(readlink $i) $(basename $i); done

##### END OF 1 #####
create_experiment(experiment_name="test")
##### END OF 1 #####

##### END OF 1 #####
from graph_cl.cli.data.raw import create_samples

create_samples(dataset_name=dataset_name, debug=True)
##### END OF 1 #####


##### START OF 1 #####
from graph_cl.cli.preprocess.build_graph import build_concept_graph_from_paths

ps = ProjectSettings(dataset_name=dataset_name)

samples = [Sample.from_pickle(p) for p in ps.samples_dir.glob("*.pkl")]

concept_names = [p.stem for p in ps.concepts_dir.glob("*.yaml")]
for concept_name in concept_names:
    for sample in samples:
        build_concept_graph_from_paths(
            sample_path=ps.get_sample_path(sample.name),
            concept_config_path=ps.get_concept_path(concept_name),
            concept_graph_path=ps.get_concept_graph_path(concept_name, sample.name),
        )
##### END OF 1 #####


##### END OF 1 #####
from graph_cl.data_models.Data import DataConfig
from graph_cl.cli.preprocess.encode_target import encode_target
from graph_cl.cli.preprocess.filter import filter_samples
from graph_cl.cli.preprocess.split import split_samples

experiment_name = "test"
ps = ProjectSettings(dataset_name=dataset_name, experiment_name=experiment_name)
ps.init()

samples = [Sample.from_pickle(p) for p in ps.samples_dir.glob("*.pkl")]

import torch

for concept_name in concept_names:
    for sample in samples:
        sample.concept_graphs[concept_name] = torch.load(
            ps.get_concept_graph_path(concept_name, sample.name)
        )

# note: create config files
data_config = DataConfig.from_yaml(ps.data_config_path)
# note: encode on all samples
samples, num_cls = encode_target(samples, data_config)
samples = filter_samples(samples, data_config)
split_info = split_samples(samples, data_config)
split_info.to_parquet(ps.split_info_path)
##### END OF 1 #####

##### END OF 1 #####
ps = ProjectSettings(dataset_name=dataset_name, experiment_name=experiment_name)
split_info = pd.read_parquet(str(ps.split_info_path))
stages = split_info.split.unique()
splits = {
    stage: [
        Sample.from_pickle(ps.get_sample_path(s))
        for s in split_info.set_index("stage")["sample_name"].loc[stage]
    ]
    for stage in stages
}
for concept_name in concept_names:
    for _, samples in splits.items():
        encode_target(samples, data_config)
        for sample in samples:
            sample.concept_graphs[concept_name] = torch.load(
                ps.get_concept_graph_path(concept_name, sample.name)
            )

from graph_cl.data_models.Model import ModelGNNConfig
from graph_cl.data_models.Train import TrainConfig
from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.train.lightning import LitGNN
from graph_cl.train.train import train
from graph_cl.dataloader.ConceptDataModuleNew import ConceptDataModule

ps = ProjectSettings(
    dataset_name=dataset_name,
    experiment_name=experiment_name,
    model_name=LitGNN.__name__,
)

dm = ConceptDataModule(splits=splits, concepts="concept_1", config=data_config)

model_config = ModelGNNConfig.from_yaml(ps.model_gnn_config_path)
model_config.num_classes = split_info.target.nunique()
model_config.in_channels = dm.num_features

train_config = TrainConfig.from_yaml(ps.pretrain_config_path)
train_config.tracking.checkpoint_dir = ps.model_dir / f"{concept_name}"

model = GNN_plus_MPL(model_config.dict())
module = LitGNN(model, config=train_config)
train(module, dm, train_config)
##### END OF 1 #####


##### END OF 1 #####
##### END OF 1 #####


##### END OF 1 #####
##### END OF 1 #####
