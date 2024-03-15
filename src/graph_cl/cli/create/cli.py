import click


@click.command()
def project():
    from .cmds import project

    project()


@click.command()
@click.option(
    "--dataset_name", "-d", required=True, type=str, help="Name of the dataset"
)
def dataset(dataset_name: str):
    from .cmds import dataset

    dataset(dataset_name=dataset_name)


@click.command()
@click.option(
    "--experiment_name", "-e", required=True, type=str, help="Name of the experiment"
)
def experiment(experiment_name: str):
    from .cmds import experiment

    experiment(experiment_name=experiment_name)
