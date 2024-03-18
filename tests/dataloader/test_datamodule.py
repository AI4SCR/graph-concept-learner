import pandas as pd

from src.graph_cl.dataloader.ConceptDataModule import ConceptDataModule
from graph_cl.data_models.Data import DataConfig
from graph_cl.data_models.Experiment import GCLExperiment
from graph_cl.data_models.Sample import Sample


def test_ConceptDataModule():
    experiment_name = "exp_1"
    factory = GCLExperiment(experiment_name=experiment_name)

    split_info = pd.read_parquet(factory.split_info_path).set_index("stage")
    splits = {
        # note: here we load from the dataset samples, i.e. with undefined split attribute
        # stage: [Sample.model_validate_from_file(x.sample_url) for _, x in split_info.loc[stage].iterrows()]
        # note: here we load from the experiment samples, created when dataset is split for the experiment
        stage: [
            Sample.model_validate_from_json(factory.get_sample_path(x.sample_name))
            for _, x in split_info.loc[stage].iterrows()
        ]
        for stage in split_info.index.unique()
    }

    data_config = DataConfig.model_validate_from_json(factory.data_config_path)
    dm = ConceptDataModule(
        splits=splits,
        model_name="gnn",
        concepts="concept_1",
        config=data_config,
        factory=factory,
        force_attr_computation=True,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    ds = dm.train_dataloader()
    batch = next(iter(ds))
    assert batch

    dm.setup(stage="val")
    ds = dm.val_dataloader()
    batch = next(iter(ds))
    assert batch

    # dm.setup(stage="test")
    # ds = dm.test_dataloader()
    # batch = next(iter(ds))
    # assert batch
