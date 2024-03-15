import pytest
from graph_cl.cli.experiment.cmd import preprocess_samples


def test_preprocess_samples():
    preprocess_samples(experiment_name="test")
    assert True


@pytest.mark.parametrize("concept_name", ["concept_1", "concept_2", "concept_3"])
def test_pretrain(concept_name):
    from graph_cl.cli.experiment.cmd import pretrain

    pretrain(experiment_name="test", concept_name=concept_name)
    assert True


def test_train():
    from graph_cl.cli.experiment.cmd import train

    train(experiment_name="test")
    assert True
