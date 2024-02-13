from pathlib import Path
from graph_cl.graph_builder.attribute_graph import attribute_graph_from_files


def test_attribute_graph():
    attribute_config_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/attribute.yaml"
    )
    graph_path = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs/knn/BaselTMA_SP41_100_X15Y5.pt"
    )
    data_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
    )
    output_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/attribute.yaml"
    )
    attribute_graph_from_files(graph_path, data_dir, attribute_config_path, output_dir)
