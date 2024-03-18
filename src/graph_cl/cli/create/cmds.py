import os
import shutil
from pathlib import Path

from ai4bmr_core.log.log import logger


def project():
    from ...data_models.Project import project

    project.create_folder_hierarchy()

    src_dir = Path(os.path.dirname(__file__))
    for config in (src_dir / "templates" / "concepts").glob("*.yaml"):
        shutil.copy(config, project.concepts_dir)

    logger.info(f"Project initialized at {project.base_dir}. ")


def experiment(experiment_name: str):
    from ...data_models.Experiment import GCLExperiment
    from ...data_models.Project import project

    experiment = GCLExperiment(name=experiment_name, project=project)
    experiment.create_folder_hierarchy()

    src_dir = Path(os.path.dirname(__file__))
    for config in (src_dir / "templates" / "configs").glob("*.yaml"):
        shutil.copy(config, experiment.configs_dir)
