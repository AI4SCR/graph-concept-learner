import click


@click.group()
def cli():
    pass


from .cmds.data.raw import data

cli.add_command(data)

from .cmds.preprocessing import preprocessing

cli.add_command(preprocessing)
