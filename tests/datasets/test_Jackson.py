import pandas as pd

from graph_cl.datasets.Jackson import Jackson
from pathlib import Path
from graph_cl.data_models.PathFactory import PathFactory


def test_Jackson():
    factory = PathFactory(dataset_name=Jackson.name)
    ds = Jackson(factory=factory)
    samples = ds.load()
    assert samples is not None
