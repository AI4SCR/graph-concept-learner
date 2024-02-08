import pandas as pd
import pickle
import click
from graph_cl.preprocessing.graphs import create_graphs_from_files


@click.command()
@click.option("--so_path", type=click.Path(exists=True), help="Path to so.pkl file")
@click.option("--valid_cores_path", type=click.Path(), help="File with valid cores")
@click.option(
    "--config_dir",
    type=click.Path(exists=True),
    help="Path to concept configs directory",
)
@click.option("--output_dir", type=click.Path(), help="Path to output directory")
def main(so_path, valid_cores_path, config_dir, output_dir):
    with open(so_path, "rb") as f:
        so = pickle.load(f)

    valid_cores = pd.read_csv(valid_cores_path, index_col="core")
    for i, core in enumerate(valid_cores.index):
        print(f"({i}/{len(valid_cores)}) {core}")
        create_graphs_from_files(
            so=so, core=core, output_dir=output_dir, concept_config_dir=config_dir
        )
