def test_split():
    from graph_cl.preprocessing.split import split_samples
    from graph_cl.data_models.Sample import Sample
    from graph_cl.data_models.Data import DataConfig
    from graph_cl.preprocessing.encode import encode_target
    from graph_cl.data_models.ExperimentPathFactory import ExperimentPathFactory
    from graph_cl.data_models.DatasetPathFactory import DatasetPathFactory

    # TODO: we cannot hardcode the experiment name
    experiment_name = "exp_1"
    factory = ExperimentPathFactory(experiment_name=experiment_name)
    factory.init()

    data_config = DataConfig.from_yaml(factory.data_config_path)
    ds = DatasetPathFactory(dataset_name=data_config.dataset_name)

    samples = [
        Sample.model_validate_from_file(p) for p in ds.samples_dir.glob("*.json")
    ]
    samples = encode_target(samples, data_config)
    split_samples(samples, data_config)
