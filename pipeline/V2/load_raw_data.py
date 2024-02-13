from graph_cl.datasets.RawDataLoaderV2 import RawDataLoader
from pathlib import Path

if __name__ == "__main__":
    raw_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/01_raw"
    )
    processed_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
    )
    loader = RawDataLoader(raw_dir=raw_dir, processed_dir=processed_dir)
    loader.load()
