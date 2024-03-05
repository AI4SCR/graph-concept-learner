from graph_cl.cli.model.pretrain import pretrain
from click.testing import CliRunner


def test_pretrain():
    runner = CliRunner()

    data_dir = "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/"
    experiment_dir = "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2"
    concept_name = "concept_1"

    result = runner.invoke(
        pretrain,
        [data_dir, experiment_dir, concept_name],
    )
    assert result.exit_code == 0
