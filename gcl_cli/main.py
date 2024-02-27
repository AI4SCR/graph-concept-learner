import click


@click.group()
def cli():
    pass


from .cmds.data.raw import data

cli.add_command(data)

from .cmds.preprocess import preprocess

cli.add_command(preprocess)


@click.group()
def model():
    pass


from gcl_cli.cmds.pretrain import pretrain

model.add_command(pretrain)

cli.add_command(model)

# %%
if __name__ == "__main__":
    cli()
