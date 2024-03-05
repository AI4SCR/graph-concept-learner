from graph_cl.datasets.Jackson import Jackson
from pathlib import Path


def test_Jackson():
    root = Path("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data")
    raw_dir = Path(root / "01_raw")
    processed_dir = Path(root / "02_processed")
    ds = Jackson(raw_dir=raw_dir, processed_dir=processed_dir)
    samples = ds.load()
    assert samples is not None
