import click


@click.group()
def data():
    pass


from .raw import jackson

data.add_command(jackson)
