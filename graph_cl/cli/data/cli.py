import click

from graph_cl.datasets.Jackson import Jackson
from pathlib import Path
import logging


@click.group()
def data():
    pass


@data.command()
@click.option(
    "raw_dir", type=click.Path(exists=True, resolve_path=True, path_type=Path)
)
@click.option("processed_dir", type=click.Path(resolve_path=True, path_type=Path))
@click.option(
    "--remove",
    is_flag=True,
    default=True,
    help="remove masks for which no metadata is available",
)
def jackson(dataset_name: str, raw_dir: Path, processed_dir: Path, remove: bool = True):
    from ...data_models.ProjectSettings import ProjectSettings

    ps = ProjectSettings(dataset_name=Jackson.name)
    ds = Jackson(raw_dir=ps.raw_dir, processed_dir=ps.processed_dir)
    df_samples_path = ps.processed_dir / "samples.parquet"
    if df_samples_path.exists():
        import pandas as pd

        df_samples = pd.read_parquet(df_samples_path)
    else:
        df_samples = ds.load()
        df_samples = df_samples.dropna()
        df_samples.astype(str).to_parquet(df_samples_path)

        from graph_cl.preprocessing.harmonize import harmonize_index

        for sample in df_samples.index:
            harmonize_index(
                mask_path=df_samples.loc[sample].mask_path,
                expr_path=df_samples.loc[sample].obs_expr_path,
                labels_path=df_samples.loc[sample].obs_labels_path,
                loc_path=df_samples.loc[sample].obs_loc_path,
                spat_path=df_samples.loc[sample].obs_spatial_path,
            )
    spls_remove = df_samples[df_samples.isna().any(axis=1)].index
    df_samples = df_samples.iloc[:10]
    samples = ds.load_samples(df_samples)
    for sample in samples:
        sample.to_pickle(ps.samples_dir / f"{sample.name}.pkl")
    loader = Jackson(raw_dir=raw_dir, processed_dir=processed_dir)
    samples = loader.load()

    # note: if any metadata is missing, remove all data for this sample
    samples = samples.dropna(axis=0)

    if remove and len(spls_remove):
        for sample in spls_remove:
            files = processed_dir.rglob(f"{sample}*")
            for file in files:
                logging.info(f"removing {file}")
                file.unlink()

    samples.astype(str).to_parquet(processed_dir / "samples.parquet")
