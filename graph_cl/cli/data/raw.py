import click

from graph_cl.datasets.RawDataset import RawDataset
from pathlib import Path
from graph_cl.preprocessing.harmonize import harmonize_index
import logging


@click.group()
def data():
    pass


@data.command()
@click.argument(
    "raw_dir", type=click.Path(exists=True, resolve_path=True, path_type=Path)
)
@click.argument("processed_dir", type=click.Path(resolve_path=True, path_type=Path))
@click.option(
    "--remove",
    is_flag=True,
    default=True,
    help="remove masks for which no metadata is available",
)
def jackson(raw_dir: Path, processed_dir: Path, remove: bool = True):
    loader = RawDataset(raw_dir=raw_dir, processed_dir=processed_dir)
    loader.load()

    cores = (processed_dir / "masks").glob("*.tiff")
    for core in cores:
        remove_core = False
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
            remove_core = True

        if loc_path.exists() is False:
            logging.info(f"{core}: Location path does not exist")
            remove_core = True

        if expr_path.exists() is False:
            logging.info(f"{core}: Expression path does not exist")
            remove_core = True

        if spat_path.exists() is False:
            logging.info(f"{core}: Spatial path does not exist")
            remove_core = True

        # note: if any metadata is missing, remove all data for this core
        if remove and remove_core:
            files = processed_dir.rglob(f"{core_name}*")
            for file in files:
                logging.info(f"removing {file}")
                file.unlink()
            continue

        harmonize_index(
            mask_path=mask_path,
            expr_path=expr_path,
            labels_path=labels_path,
            loc_path=loc_path,
            spat_path=spat_path,
        )
