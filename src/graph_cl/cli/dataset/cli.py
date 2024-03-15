import click


@click.command()
@click.option("--dataset_name", "-d", type=str)
def download(dataset_name: str):
    from .cmds import download

    download(dataset_name=dataset_name)


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
    from .cmds import process

    process(dataset_name=dataset_name)
