import click


# NOTE: interface if experiment_name is defined on the group
# @click.command(help="Preprocess data according to data.yaml configuration.")
# @click.pass_obj
# def preprocess(ctx):
#     print(ctx['experiment_name'])


@click.command(help="Concept graph computations, according to data.yaml configuration")
@click.option(
    "--experiment_name",
    "-e",
    required=True,
    type=str,
    default="jackson",
    show_default=True,
    help="Name of the experiment the sample belongs to",
)
@click.option("--sample_name", "-s", required=True, type=str, help="Name of the sample")
@click.option(
    "--concept_name",
    "-c",
    required=True,
    type=str,
    help="Name of the concept yaml file",
)
@click.option(
    "--force", "-f", is_flag=True, help="Force overwrite of existing concept graph"
)
def create_concept_graph(
    experiment_name: str, sample_name: str, concept_name: str, force: bool
) -> None:
    from .cmds import create_concept_graph

    create_concept_graph(
        experiment_name=experiment_name,
        sample_name=sample_name,
        concept_name=concept_name,
        force=force,
    )


@click.command(
    help="Preprocess samples and split according to data.yaml configuration."
)
@click.option(
    "--experiment_name", "-e", required=True, type=str, help="Name of the experiment"
)
@click.option("--verbose", "-v", count=True, default=0, help="Print verbose output")
def preprocess(experiment_name: str, verbose: int):
    from .cmds import preprocess_samples

    preprocess_samples(experiment_name=experiment_name, verbose=verbose)


@click.command(help="Pretrain GNN model on a concept.")
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
    from .cmds import pretrain

    pretrain(experiment_name=experiment_name, concept_name=concept_name)


@click.command(help="Train the complete graph concept learner model.")
@click.option(
    "--experiment_name", "-e", required=True, type=str, help="Name of the experiment"
)
def train(experiment_name: str):
    from .cmds import train

    train(experiment_name=experiment_name)
