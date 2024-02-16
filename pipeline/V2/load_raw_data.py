from graph_cl.datasets.RawDataLoaderV2 import RawDataLoader
from pathlib import Path
from graph_cl.preprocessing.filterV2 import harmonize_index
import logging

if __name__ == "__main__":
    raw_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/01_raw"
    )
    processed_dir = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
    )
    loader = RawDataLoader(raw_dir=raw_dir, processed_dir=processed_dir)
    loader.load()

    cores = (processed_dir / "masks").glob("*.tiff")
    for core in cores:
        core_name = core.stem
        mask_path = processed_dir / "masks" / f"{core_name}.tiff"
        labels_path = processed_dir / "labels" / "observations" / f"{core_name}.parquet"
        loc_path = (
            processed_dir
            / "features"
            / "observations"
            / "location"
            / f"{core_name}.parquet"
        )
        expr_path = (
            processed_dir
            / "features"
            / "observations"
            / "expression"
            / f"{core_name}.parquet"
        )
        spat_path = (
            processed_dir
            / "features"
            / "observations"
            / "spatial"
            / f"{core_name}.parquet"
        )

        if labels_path.exists() is False:
            logging.info(f"{core}: Labels path does not exist")
            continue
        if loc_path.exists() is False:
            logging.info(f"{core}: Location path does not exist")
            continue
        if expr_path.exists() is False:
            logging.info(f"{core}: Expression path does not exist")
            continue
        if spat_path.exists() is False:
            logging.info(f"{core}: Spatial path does not exist")
            continue

        harmonize_index(
            mask_path=mask_path,
            expr_path=expr_path,
            labels_path=labels_path,
            loc_path=loc_path,
            spat_path=spat_path,
        )
