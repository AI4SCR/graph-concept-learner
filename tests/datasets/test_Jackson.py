import pandas as pd

from graph_cl.datasets.Jackson import Jackson
from pathlib import Path
from graph_cl.data_models.DatasetPathFactory import DatasetPathFactory


def test_Jackson():
    factory = DatasetPathFactory(dataset_name=Jackson.name)
    ds = Jackson(factory=factory)
    samples = ds.process()
    assert samples is not None
