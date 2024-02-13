from graph_cl.preprocessing.filterV2 import harmonize_index
from pathlib import Path


def test_filter_samples():
    root = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
    )
    core = "BaselTMA_SP41_100_X15Y5"
    mask_path = root / "masks" / f"{core}.tiff"
    labels_path = root / "labels" / "observations" / f"{core}.parquet"
    loc_path = root / "features" / "observations" / "locations" / f"{core}.parquet"
    expr_path = root / "features" / "observations" / "expression" / f"{core}.parquet"
    spat_path = root / "features" / "observations" / "spatial" / f"{core}.parquet"
    harmonize_index(
        mask_path=mask_path,
        expr_path=expr_path,
        labels_path=labels_path,
        loc_path=loc_path,
        spat_path=spat_path,
    )
