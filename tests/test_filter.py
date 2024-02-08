from graph_cl.preprocessing.filter import filter_samples
from graph_cl import CONFIG

experiment_dir = CONFIG.project.root / "experiments" / CONFIG.experiment.name

so_path = CONFIG.data.root / "01_raw" / "so.pkl"
concepts_dir = experiment_dir / "configuration" / "concept_configs"
output_dir = experiment_dir / "data" / "valid_cores.csv"
target = CONFIG.experiment.target
min_cells_per_graph = CONFIG.data.processing.filter.min_cells_per_graph
print(output_dir)
filter_samples(so_path, concepts_dir, target, min_cells_per_graph, output_dir)
