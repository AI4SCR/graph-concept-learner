import click


@click.group()
def model():
    pass


from .pretrain import pretrain
from .train import train

model.add_command(pretrain)
model.add_command(train)
