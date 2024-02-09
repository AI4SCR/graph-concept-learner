from graph_cl.train.pretrain import pretrain_concept, pretrain_concept_from_files
from graph_cl import CONFIG

experiment_root = CONFIG.project.root / CONFIG.experiment.name
config_root = experiment_root / "configuration"
train_config_path = config_root / "training.yaml"
model_config_path = config_root / "model.yaml"
fold_meta_data_path = experiment_root / "data" / "folds" / "fold_0.csv"
root = experiment_root / "data" / "graphs" / "knn"

pretrain_concept_from_files(
    root, fold_meta_data_path, model_config_path, train_config_path
)
