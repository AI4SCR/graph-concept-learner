import pandas as pd

from graph_cl.datasets.Jackson import Jackson
from pathlib import Path


def test_Jackson():
    root = Path("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data")
    raw_dir = Path(root / "01_raw")
    processed_dir = Path(root / "02_processed")
    ds = Jackson(raw_dir=raw_dir, processed_dir=processed_dir)
    samples = ds.load()
    assert samples is not None


def test_Jackson_load_samples():
    root = Path("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data")
    raw_dir = Path(root / "01_raw")
    processed_dir = Path(root / "02_processed")
    samples_path = root / "02_processed" / "samples.parquet"
    ds = Jackson(raw_dir=raw_dir, processed_dir=processed_dir)
    df_samples = pd.read_parquet(samples_path)
    samples = ds.load_samples(df_samples)
    assert len(samples)
