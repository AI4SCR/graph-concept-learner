from graph_cl.train.pretrain import pretrain_concept, pretrain_concept_from_files
from pathlib import Path

fold_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
)
model_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model.yaml"
)
pretrain_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/pretrain.yaml"
)


def test_pretrain_concept():
    pretrain_concept_from_files(
        concept="knn_all",
        fold_path=fold_path,
        model_config_path=model_config_path,
        pretrain_config_path=pretrain_config_path,
    )
