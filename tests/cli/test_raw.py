from graph_cl.cli.data.raw import jackson
from click.testing import CliRunner


def test_jackson():
    runner = CliRunner()

    raw_dir = "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/01_raw"
    processed_dir = (
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
    )

    result = runner.invoke(
        jackson,
        [
            str(raw_dir),
            str(processed_dir),
        ],
    )
    assert result.exit_code == 0
