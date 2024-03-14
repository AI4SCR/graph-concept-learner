import pandas as pd

from graph_cl.dataloader.ConceptDataModule import ConceptDataModule
from graph_cl.data_models.Data import DataConfig
from pathlib import Path
import yaml
from graph_cl.data_models.ProjectSettings import ProjectSettings
from graph_cl.data_models.Sample import Sample


def test_ConceptDataModule():
    ps = ProjectSettings(
        dataset_name="jackson",
        experiment_name="test",
        model_name="my_model",
    )

    split_info = pd.read_parquet(str(ps.split_info_path))
    stages = set(split_info.split)

    splits = {
        stage: [
            Sample.from_pickle(ps.get_sample_path(s))
            for s in split_info.set_index("stage")["sample_name"].loc[stage]
        ]
        for stage in stages
    }

    data_config = DataConfig.from_yaml(ps.data_config_path)

    dm = ConceptDataModule(
        splits=splits,
        concepts="concept_1",
        config=data_config,
        save_samples_dir=ps.experiment_samples_dir,
        save_dataset_dir=ps.experiment_dataset_dir,
        save_attributes_dir=ps.experiment_attributes_dir,
    )
    dm.setup(stage="fit")
