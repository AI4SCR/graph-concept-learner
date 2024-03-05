import click


@click.group()
def preprocess():
    pass


from .build_graph import build_concept_graph

preprocess.add_command(build_concept_graph)
