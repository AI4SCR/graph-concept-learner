import click


@click.command()
@click.option("--sample_name", "-s", required=True, type=str, help="Name of the sample")
@click.option(
    "--dataset_name",
    "-d",
    required=True,
    type=str,
    default="jackson",
    show_default=True,
    help="Name of the dataset the sample belongs to",
)
@click.option(
    "--concept_name",
    "-c",
    required=True,
    type=str,
    help="Name of the concept yaml file",
)
def create(sample_name: str, dataset_name: str, concept_name: str):
    from .cmds import create_concept_graph

    create_concept_graph(
        sample_name=sample_name, dataset_name=dataset_name, concept_name=concept_name
    )
