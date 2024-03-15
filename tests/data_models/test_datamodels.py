import pathlib
import tempfile

from graph_cl.data_models.ProjectPathFactory import ProjectPathFactory
from graph_cl.data_models.ExperimentPathFactory import ExperimentPathFactory
from graph_cl.data_models.DatasetPathFactory import DatasetPathFactory


def test_ProjectPathFactory():
    import tempfile
    from pathlib import Path

    factory = ProjectPathFactory()
    assert factory.model_dump_json()
    assert factory.get_concept_config_path("concept_1")

    with tempfile.TemporaryDirectory() as tmpdir:
        factory = ProjectPathFactory(data_dir=tmpdir)
        assert isinstance(factory.data_dir, Path)
        assert str(factory.data_dir) == tmpdir

        tmpdir = Path(tmpdir)
        factory.init()
        assert (tmpdir / "datasets").exists()
        assert (tmpdir / "concepts").exists()
        assert (tmpdir / "experiments").exists()


def test_DataPathFactory():
    import tempfile
    from pathlib import Path

    factory = DatasetPathFactory(dataset_name="test")
    assert factory.model_dump_json()

    with tempfile.TemporaryDirectory() as tmpdir:
        factory = DatasetPathFactory(data_dir=tmpdir, dataset_name="test")
        factory.init()

        tmpdir = Path(tmpdir)
        assert (tmpdir / "datasets").exists()
        assert (tmpdir / "concepts").exists()
        assert (tmpdir / "experiments").exists()
        assert (tmpdir / "datasets" / "test").exists()
        assert (tmpdir / "datasets" / "test" / "01_raw").exists()
        assert (tmpdir / "datasets" / "test" / "02_processed").exists()
        assert (tmpdir / "datasets" / "test" / "03_concept_graphs").exists()
        assert (tmpdir / "datasets" / "test" / "04_samples").exists()


def test_ExperimentPathFactory():
    import tempfile
    from pathlib import Path

    factory = ExperimentPathFactory(experiment_name="test")
    assert factory.model_dump_json()

    with tempfile.TemporaryDirectory() as tmpdir:
        factory = ExperimentPathFactory(data_dir=tmpdir, experiment_name="test")
        factory.init()

        tmpdir = Path(tmpdir)
        assert (tmpdir / "datasets").exists()
        assert (tmpdir / "concepts").exists()
        assert (tmpdir / "experiments").exists()
        assert (tmpdir / "experiments" / "test").exists()
        assert (tmpdir / "experiments" / "test" / "00_configuration").exists()
        assert (tmpdir / "experiments" / "test" / "05_attributes").exists()
        assert (tmpdir / "experiments" / "test" / "06_samples").exists()
        assert (tmpdir / "experiments" / "test" / "07_datasets").exists()
        assert (tmpdir / "experiments" / "test" / "08_models").exists()
        assert (tmpdir / "experiments" / "test" / "09_predictions").exists()
