# note: create config files
data_config = DataConfig.from_yaml(ps.data_config_path)
# note: encode on all samples
samples, num_cls = encode_target(samples, data_config)
samples = filter_samples(samples, data_config)
split_info = split_samples(samples, data_config)
split_info.to_parquet(ps.split_info_path)
##### END OF 1 #####

from graph_cl.data_models.Model import ModelGNNConfig
from graph_cl.data_models.Train import TrainConfig
from graph_cl.models.gnn import GNN_plus_MPL
from graph_cl.train.lightning import LitGNN
from graph_cl.train.train import train
from graph_cl.dataloader.ConceptDataModule import ConceptDataModule

ps = ProjectSettings(
    dataset_name=dataset_name,
    experiment_name=experiment_name,
    model_name=LitGNN.__name__,
)

dm = ConceptDataModule(
    splits=splits,
    concepts="concept_1",
    config=data_config,
    save_samples_dir=ps.experiment_samples_dir,
    save_dataset_dir=ps.experiment_dataset_dir,
    save_attributes_dir=ps.experiment_attributes_dir,
)

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
