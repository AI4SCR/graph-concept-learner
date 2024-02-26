from graph_cl.preprocessing.split import create_folds
from graph_cl import CONFIG

experiment_dir = CONFIG.project.root / "experiments" / CONFIG.experiment.name

method = CONFIG.data.processing.split.method
valid_samples_path = experiment_dir / "data" / "valid_cores.csv"
output_dir = experiment_dir / "data" / "folds"
n_folds = CONFIG.data.processing.split.n_folds
train_size = CONFIG.data.processing.split.train_size

create_folds(
    method=method,
    valid_samples_path=valid_samples_path,
    n_folds=n_folds,
    train_size=train_size,
    output_dir=output_dir,
)
