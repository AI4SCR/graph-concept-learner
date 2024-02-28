from pathlib import Path
from graph_cl.train.gcl import train_from_files

fold_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
)
concept_configs_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/00_concepts"
)

train_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/train.yaml"
)
model_gcl_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gcl.yaml"
)
model_gnn_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gnn.yaml"
)
train_from_files(
    fold_dir=fold_dir,
    train_config_path=train_config_path,
    model_gcl_config_path=model_gcl_config_path,
    model_gnn_config_path=model_gnn_config_path,
)
