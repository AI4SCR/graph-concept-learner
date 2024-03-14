import click
from .cmds import download_dataset, process_dataset


@click.command()
@click.option("--dataset_name", "-d", type=str)
def download(dataset_name: str):
    download_dataset(dataset_name=dataset_name)


@click.command()
@click.option(
    "--dataset_name",
    "-d",
    required=True,
    type=str,
    default="jackson",
    show_default=True,
)
def process(dataset_name):
    process_dataset(dataset_name=dataset_name)
