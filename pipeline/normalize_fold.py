import click
from graph_cl.preprocessing.normalize import normalize_fold


@click.command()
@click.argument("so_path", type=click.Path(), help="Path raw so object file")
@click.argument("fold_path", type=click.Path(), help="Path to directory fold")
@click.argument("output_dir", type=click.Path(), help="Path to output directory")
@click.argument("config_path", type=click.Path(), help="Path to configuration file")
def main(so_path, fold_path, output_dir, config_path):
    normalize_fold(so_path, fold_path, output_dir, config_path)
