from graph_cl.train.pretrain import pretrain_concept_from_files
from pathlib import Path
from graph_cl.cli.model import pretrain


def test_pretrain_concept():
    fold_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
    )
    model_config_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model.yaml"
    )
    pretrain_config_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/pretrain.yaml"
    )
    pretrain_concept_from_files(
        concept="knn_all",
        fold_path=fold_path,
        model_config_path=model_config_path,
        train_config_path=pretrain_config_path,
    )


def test_pretrain_cli():
    from click.testing import CliRunner

    runner = CliRunner()
    folds_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/"
    )
    fold_path = folds_dir / "fold_0"

    model_config_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gnn.yaml"
    )
    train_config_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/pretrain.yaml"
    )

    runner.invoke(
        pretrain,
        [
            "radius_tumor_immune",
            str(fold_path),
            str(model_config_path),
            str(train_config_path),
        ],
    )
