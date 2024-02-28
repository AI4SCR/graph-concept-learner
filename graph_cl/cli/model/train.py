from graph_cl.train.gcl import train_from_files
import click
from pathlib import Path


@click.command()
@click.argument("fold_dir", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.argument(
    "train_config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "model_gcl_config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "model_gnn_config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def train(
    fold_dir: Path,
    train_config_path: Path,
    model_gcl_config_path: Path,
    model_gnn_config_path: Path,
):
    train_from_files(
        fold_dir, train_config_path, model_gcl_config_path, model_gnn_config_path
    )
