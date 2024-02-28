from pathlib import Path
from graph_cl.cli.model.train import train
from click.testing import CliRunner


def test_model_train():
    runner = CliRunner()

    fold_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
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

    result = runner.invoke(
        train,
        [
            str(fold_dir),
            str(train_config_path),
            str(model_gcl_config_path),
            str(model_gnn_config_path),
        ],
    )
    assert result.exit_code == 0
