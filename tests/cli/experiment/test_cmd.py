from graph_cl.cli.experiment.cmd import preprocess_samples


def test_preprocess_samples():
    preprocess_samples(experiment_name="test")
    assert True
