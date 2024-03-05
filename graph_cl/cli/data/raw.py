import click

from graph_cl.datasets.Jackson import Jackson
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
    loader = Jackson(raw_dir=raw_dir, processed_dir=processed_dir)
    samples = loader.load()

    # note: if any metadata is missing, remove all data for this sample
    spls_remove = samples[samples.isna().any(axis=1)].index
    samples = samples.dropna(axis=0)

    if remove and len(spls_remove):
        for sample in spls_remove:
            files = processed_dir.rglob(f"{sample}*")
            for file in files:
                logging.info(f"removing {file}")
                file.unlink()

    samples.astype(str).to_parquet(processed_dir / "samples.parquet")
    samples.astype(str).to_csv(processed_dir / "samples.csv")

    for sample in samples.index:
        harmonize_index(
            mask_path=samples.loc[sample].mask_path,
            expr_path=samples.loc[sample].obs_expr_path,
            labels_path=samples.loc[sample].obs_labels_path,
            loc_path=samples.loc[sample].obs_loc_path,
            spat_path=samples.loc[sample].obs_spatial_path,
        )
