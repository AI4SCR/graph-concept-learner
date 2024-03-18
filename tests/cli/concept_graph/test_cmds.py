from graph_cl.cli.concept_graph.cmds import create_concept_graph


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
