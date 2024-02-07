import click
from graph_cl.preprocessing.split import create_folds


@click.command()
@click.argument("method", type=str)
@click.argument(
    "valid_samples_path",
    type=click.Path(exists=True),
    help="Path to file with valid samples",
)
@click.argument("n_folds", type=int, help="Number of folds")
@click.argument("train_size", type=float, help="Train size")
@click.argument("output_dir", type=click.Path(), help="Path to output directory")
def main(method, valid_samples_path, n_folds, train_size, output_dir):
    create_folds(method, valid_samples_path, n_folds, train_size, output_dir)
