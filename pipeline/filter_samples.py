import click
from graph_cl.preprocessing.filter import filter_samples


@click.command()
@click.argument("so_path", type=click.Path(exists=True), help="Path raw so.pkl")
@click.argument(
    "concepts_dir", type=click.Path(exists=True), help="Path to concept configs"
)
@click.argument("target", type=str, help="Target label")
@click.argument(
    "min_cells_per_graph", type=int, help="Minimum number of cells per graph"
)
@click.argument(
    "output", type=click.Path(), help="Path to output file with valid cores"
)
def main(so_path, concepts_dir, target: str, min_cells_per_graph: int, output):
    filter_samples(so_path, concepts_dir, target, min_cells_per_graph, output)
