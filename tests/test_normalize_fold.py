from graph_cl.preprocessing.normalize import normalize_fold
from graph_cl import CONFIG
from graph_cl.configuration import config_path


experiment_dir = CONFIG.project.root / "experiments" / CONFIG.experiment.name

so_path = CONFIG.data.root / "01_raw" / "so.pkl"
concepts_dir = experiment_dir / "configuration" / "concept_configs"
fold_path = experiment_dir / "data" / "folds" / "fold_0.csv"
fold_path_1 = experiment_dir / "data" / "folds" / "fold_1.csv"
fold_path_2 = experiment_dir / "data" / "folds" / "fold_2.csv"
output_dir = experiment_dir / "data" / "folds_normalized"

normalize_fold(
    so_path=so_path, fold_path=fold_path, output_dir=output_dir, config_path=config_path
)
normalize_fold(
    so_path=so_path,
    fold_path=fold_path_1,
    output_dir=output_dir,
    config_path=config_path,
)
normalize_fold(
    so_path=so_path,
    fold_path=fold_path_2,
    output_dir=output_dir,
    config_path=config_path,
)
