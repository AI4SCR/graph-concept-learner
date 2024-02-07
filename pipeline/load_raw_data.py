import click
from graph_cl.datasets.RawDataLoader import RawDataLoader


@click.command()
@click.argument("raw_dir", help="Path to the raw data directory")
@click.argument("output", help="Path to the created so.pkl file")
def main(raw_dir, output):
    loader = RawDataLoader(raw_dir=raw_dir, output=output)
    loader.create_so()
