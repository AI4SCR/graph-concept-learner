from graph_cl import CONFIG

# %%
from graph_cl.datasets.RawDataLoader import RawDataLoader


def test_raw_data_loader():
    so = RawDataLoader(CONFIG)()


test_raw_data_loader()
