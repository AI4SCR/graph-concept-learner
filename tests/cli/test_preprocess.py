from graph_cl.cli.preprocess.build_graph import build_concept_graph
from click.testing import CliRunner


def test_build_graph():
    runner = CliRunner()

    mask_path = "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed/masks/BaselTMA_SP41_100_X15Y5.tiff"
    labels_path = "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed/labels/observations/BaselTMA_SP41_100_X15Y5.parquet"
    concept_name = "concept_1"
    concept_config_path = "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/00_concepts/concepts.yaml"
    output_path = f"/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs/{concept_name}/BaselTMA_SP41_100_X15Y5.tiff"

    result = runner.invoke(
        build_concept_graph,
        [
            concept_name,
            str(mask_path),
            str(labels_path),
            str(concept_config_path),
            str(output_path),
        ],
    )
    assert result.exit_code == 0
