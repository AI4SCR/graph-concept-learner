from pathlib import Path
from graph_cl.graph_builder.attribute_graph import attribute_graph_from_files

attribute_config_path = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/attribute.yaml"
)
data_dir = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
)
concept_name = "knn"
graph_path = (
    Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs"
    )
    / concept_name
    / "BaselTMA_SP41_100_X15Y5.pt"
)
output_dir = (
    Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/04_attributed_graphs/"
    )
    / concept_name
)
attribute_graph_from_files(graph_path, data_dir, attribute_config_path, output_dir)
