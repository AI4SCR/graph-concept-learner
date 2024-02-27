from click.testing import CliRunner
from gcl_cli.cmds.preprocess.attribut_graphs import attribute_graphs


def test_attribute_graphs():
    runner = CliRunner()

    from pathlib import Path

    EXPERIMENT_DIR = Path(
        "~/data/ai4src/graph-concept-learner/experiments/ERStatusV2"
    ).expanduser()
    DATA_DIR = Path("~/data/ai4src/graph-concept-learner/test/").expanduser()

    result = runner.invoke(attribute_graphs, [str(EXPERIMENT_DIR), str(DATA_DIR)])
