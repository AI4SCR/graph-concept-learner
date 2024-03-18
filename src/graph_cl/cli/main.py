import click


# %%
@click.group()
def cli():
    pass


from .create import create
from .experiment import experiment

cli.add_command(create)
cli.add_command(experiment)

# %%
if __name__ == "__main__":
    cli(obj={})
