import click


# %%
@click.group()
def cli():
    pass


from .dataset import dataset
from .concept_graph import concept_graph

cli.add_command(dataset)
cli.add_command(concept_graph)

# %%
if __name__ == "__main__":
    cli()
