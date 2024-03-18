import pytest

from graph_cl.cli.experiment.cmds import preprocess_samples, create_concept_graph


def test_preprocess_samples():
    preprocess_samples(experiment_name="exp_1")
    assert True


def test_create_concept_graph():
    sample_name = "BaselTMA_SP41_2_X2Y8"
    experiment_name = "exp_1"
    concept_name = "concept_1"
    create_concept_graph(
        experiment_name=experiment_name,
        sample_name=sample_name,
        concept_name=concept_name,
    )
    assert True


@pytest.mark.parametrize("concept_name", ["concept_1", "concept_2", "concept_3"])
def test_pretrain(concept_name):
    from graph_cl.cli.experiment.cmds import pretrain

    pretrain(experiment_name="test", concept_name=concept_name)
    assert True


def test_train():
    from graph_cl.cli.experiment.cmds import train

    train(experiment_name="exp_1")
    assert True
