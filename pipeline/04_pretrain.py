from graph_cl.train.pretrain import pretrain_concept_from_files
from pathlib import Path
import yaml

folds_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/"
)
data_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/data.yaml"
)
model_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gnn.yaml"
)
pretrain_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/pretrain.yaml"
)

if __name__ == "__main__":

    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)

    for fold_path in folds_dir.iterdir():
        for concept in data_config["concepts"]:
            pretrain_concept_from_files(
                concept=concept,
                fold_path=fold_path,
                model_config_path=model_config_path,
                train_config_path=pretrain_config_path,
            )
