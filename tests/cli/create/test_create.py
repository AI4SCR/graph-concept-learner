def test_create_project():
    from graph_cl.cli.create.cmds import project

    project()


def test_create_experiment():
    from graph_cl.cli.create.cmds import experiment

    experiment("test")
