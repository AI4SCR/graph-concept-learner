import click


# %%
@click.group()
def cli():
    pass


from .create import create
from .dataset import dataset
from .concept_graph import concept_graph
from .experiment import experiment

cli.add_command(create)
cli.add_command(dataset)
cli.add_command(concept_graph)
cli.add_command(experiment)

# %%
if __name__ == "__main__":
    cli(obj={})
