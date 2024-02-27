from click.testing import CliRunner
from gcl_cli.cmds.preprocess.build_graph import build_concept_graph


def test_build_concept_graph():
    runner = CliRunner()

    from pathlib import Path

    sample = "BaselTMA_SP42_145_X2Y4_142"
    MASK_PATH = Path(
        f"/Users/adrianomartinelli/data/ai4src/graph-concept-learner/test/02_processed/masks/{sample}.tiff"
    ).expanduser()
    LABELS_PATH = Path(
        f"~/data/ai4src/graph-concept-learner/test/02_processed/labels/observations/{sample}.parquet"
    ).expanduser()
    CONFIG_PATH = Path(
        "~/data/ai4src/graph-concept-learner/data/00_concepts/radius_tumor_immune.yaml"
    ).expanduser()
    OUTPUT_DIR = Path(
        "~/data/ai4src/graph-concept-learner/test/03_concept_graphs/"
    ).expanduser()

    runner.invoke(
        build_concept_graph,
        [str(MASK_PATH), str(LABELS_PATH), str(CONFIG_PATH), str(OUTPUT_DIR)],
    )
