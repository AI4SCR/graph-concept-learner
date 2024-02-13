from pathlib import Path

# %%
from graph_cl.datasets.RawDataLoaderV2 import RawDataLoader


def test_raw_data_loader():
    raw_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/01_raw"
    )
    processed_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
    )
    loader = RawDataLoader(raw_dir, processed_dir)
    loader.load()


test_raw_data_loader()
