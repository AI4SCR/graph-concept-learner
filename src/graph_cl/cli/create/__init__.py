import click


@click.group()
def create():
    pass


from .cli import project, dataset, experiment

create.add_command(project)
create.add_command(dataset)
create.add_command(experiment)
