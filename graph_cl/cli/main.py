import click
from graph_cl.cli.data import data
from graph_cl.cli.preprocess import preprocess
from graph_cl.cli.model import model

# %%
@click.group()
def cli():
    pass


cli.add_command(data)
cli.add_command(preprocess)
cli.add_command(model)

# %%
if __name__ == "__main__":
    cli()
