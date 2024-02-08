from graph_cl.preprocessing.graphs import (
    create_graph_from_files,
    create_graphs_from_files,
)
from graph_cl import CONFIG

core = "BaselTMA_SP41_57_X8Y6"
experiment_path = CONFIG.project.root / "experiments" / CONFIG.experiment.name
so_path = experiment_path / "data" / "folds_normalized" / "fold_0.pkl"
config_path = (
    experiment_path / "configuration" / "concept_configs" / "concept_1_radius.yaml"
)
attribute_config_path = (
    experiment_path / "configuration" / "attribute_configs" / "all_X_cols.yaml"
)
output_dir = experiment_path / "data" / "graphs"
valid_cores_path = experiment_path / "data" / "valid_cores.csv"

# create_graph_from_files(so_path=so_path, core=core, output_dir=output_dir, config_path=config_path)

import pandas as pd
import pickle

config_dir = experiment_path / "configuration" / "concept_configs"
valid_cores = pd.read_csv(
    experiment_path / "data" / "valid_cores.csv", index_col="core"
)
with open(so_path, "rb") as f:
    so = pickle.load(f)
for i, core in enumerate(valid_cores.index):
    print(f"({i}/{len(valid_cores)}) {core}")
    create_graphs_from_files(
        so=so,
        core=core,
        concept_config_dir=config_dir,
        attribute_config_path=attribute_config_path,
        valid_cores_path=valid_cores_path,
        output_dir=output_dir,
    )
    if i == 10:
        break
