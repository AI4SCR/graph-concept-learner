import click


# NOTE: interface if experiment_name is defined on the group
# @click.command(help="Preprocess data according to data.yaml configuration.")
# @click.pass_obj
# def preprocess(ctx):
#     print(ctx['experiment_name'])


@click.command(help="Preprocess data according to data.yaml configuration.")
@click.option(
    "--experiment_name", "-e", required=True, type=str, help="Name of the experiment"
)
def preprocess(experiment_name: str):
    from .cmd import preprocess_samples

    preprocess_samples(experiment_name=experiment_name)


@click.command(help="Preprocess data according to data.yaml configuration.")
@click.option(
    "--experiment_name", "-e", required=True, type=str, help="Name of the experiment"
)
@click.option(
    "--concept_name",
    "-c",
    required=True,
    type=str,
    help="Name of the concept to pretrain the GNN model on",
)
def pretrain(experiment_name: str, concept_name: str):
    from .cmd import pretrain

    pretrain(experiment_name=experiment_name, concept_name=concept_name)


@click.command(help="Preprocess data according to data.yaml configuration.")
@click.option(
    "--experiment_name", "-e", required=True, type=str, help="Name of the experiment"
)
def train(experiment_name: str):
    from .cmd import train

    train(experiment_name=experiment_name)
