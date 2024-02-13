from graph_cl.preprocessing.filterV2 import harmonize_index
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    root = Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed"
    )
    cores = (root / "masks").glob("*.tiff")
    for core in cores:
        core_name = core.stem
        mask_path = root / "masks" / f"{core_name}.tiff"
        labels_path = root / "labels" / "observations" / f"{core_name}.parquet"
        loc_path = (
            root / "features" / "observations" / "location" / f"{core_name}.parquet"
        )
        expr_path = (
            root / "features" / "observations" / "expression" / f"{core_name}.parquet"
        )
        spat_path = (
            root / "features" / "observations" / "spatial" / f"{core_name}.parquet"
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
