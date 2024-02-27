from graph_cl.train.pretrain import pretrain_concept_from_files
from pathlib import Path
import click


@click.command()
@click.argument("concept", type=str)
@click.argument(
    "fold_path", type=click.Path(exists=True, dir_okay=True, path_type=Path)
)
@click.argument(
    "model_config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "pretrain_config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def pretrain(
    fold_path: Path, concept: str, model_config_path: Path, pretrain_config_path: Path
):
    pretrain_concept_from_files(
        concept=concept,
        fold_path=fold_path,
        model_config_path=model_config_path,
        train_config_path=pretrain_config_path,
    )
