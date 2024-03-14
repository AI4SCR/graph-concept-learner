import click


@click.group()
def dataset():
    pass


from .cli import download, process

dataset.add_command(download)
dataset.add_command(process)
