def test_split():
    from graph_cl.preprocessing.split import split_samples
    from graph_cl.data_models.Sample import Sample
    from graph_cl.data_models.Data import DataConfig
    from graph_cl.preprocessing.encode import encode_target
    from graph_cl.data_models.Experiment import GCLExperiment
    from graph_cl.datasets.GCLDataset import DatasetPathFactory

    # TODO: we cannot hardcode the experiment name
    experiment_name = "exp_1"
    factory = GCLExperiment(experiment_name=experiment_name)
    factory.create_folder_hierarchy()

    data_config = DataConfig.model_validate_from_json(factory.data_config_path)
    ds = DatasetPathFactory(dataset_name=data_config.dataset_name)

    samples = [
        Sample.model_validate_from_json(p) for p in ds.samples_dir.glob("*.json")
    ]
    samples = encode_target(samples, data_config)
    split_samples(samples, data_config)
