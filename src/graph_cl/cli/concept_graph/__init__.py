import click


@click.group()
def concept_graph():
    pass


from .cli import create

concept_graph.add_command(create)
