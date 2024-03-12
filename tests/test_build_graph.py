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
from graph_cl.cli.preprocess import _build_graph


def test_build_concept_graph():
    from skimage.io import imread
    from graph_cl.data_models.Concept import ConceptConfig
    import yaml

    with open(CONFIG_PATH) as f:
        concept_config = yaml.load(f, Loader=yaml.Loader)
        concept_config = ConceptConfig(**concept_config)

    mask = imread(MASK_PATH, plugin="tifffile")
    _build_graph(
        mask=mask,
        topology=concept_config.graph.topology,
        params=concept_config.graph.params,
    )
