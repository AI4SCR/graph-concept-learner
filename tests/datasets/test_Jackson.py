from graph_cl.datasets.Jackson import Jackson


def test_load_cached_Jackson():
    ds = Jackson()
    assert len(ds._data) == 746


def test_Jackson_process():
    ds = Jackson(force_process=True)
    assert True
