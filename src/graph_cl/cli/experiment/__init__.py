import click

# NOTE: example of how to build a group with a parameter that can be access by the subcommands
# @click.group(help="Experiment commands")
# @click.option('--experiment_name', '-e', help='Name of the experiment')
# @click.pass_context
# def experiment(ctx, experiment_name: str):
#     ctx.ensure_object(dict)
#     ctx.obj["experiment_name"] = experiment_name


@click.group(help="Experiment commands")
def experiment():
    pass


from .cli import preprocess, pretrain, train

experiment.add_command(preprocess)
experiment.add_command(pretrain)
experiment.add_command(train)
