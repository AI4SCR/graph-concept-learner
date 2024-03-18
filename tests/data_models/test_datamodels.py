from pathlib import Path

from graph_cl.data_models.Experiment import GCLExperiment as Experiment
from graph_cl.data_models.Project import GCLProject


def test_project_env_vars():
    project = GCLProject()
    assert project.name == "graph-concept-learner"
    assert project.dir == Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data"
    )


def test_project():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        project = GCLProject(dir=tmpdir, name="test_project")
        project.create_folder_hierarchy()

        assert (tmpdir / "01_datasets").exists()
        assert (tmpdir / "02_experiments").exists()
        assert (tmpdir / "03_concepts").exists()

        # assert (tmpdir / "datasets" / "test").exists()
        # assert (tmpdir / "datasets" / "test" / "01_raw").exists()
        # assert (tmpdir / "datasets" / "test" / "02_processed").exists()
        # assert (tmpdir / "datasets" / "test" / "03_concept_graphs").exists()
        # assert (tmpdir / "datasets" / "test" / "04_samples").exists()


def test_experiment():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        project = GCLProject(dir=tmpdir)
        experiment = Experiment(name="test", project=project)
        experiment.create_folder_hierarchy()

        tmpdir = Path(tmpdir)
        assert (tmpdir / "02_experiments").exists()
        assert (tmpdir / "02_experiments" / "test").exists()
        assert (tmpdir / "02_experiments" / "test" / "00_configurations").exists()
        assert (tmpdir / "02_experiments" / "test" / "011_attributes").exists()
        assert (tmpdir / "02_experiments" / "test" / "01_samples").exists()
        assert (tmpdir / "02_experiments" / "test" / "02_models").exists()
        assert (tmpdir / "02_experiments" / "test" / "03_predictions").exists()
        assert (tmpdir / "02_experiments" / "test" / "04_results").exists()
