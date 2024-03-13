from graph_cl.datasets.Jackson import Jackson
from pathlib import Path
from graph_cl.preprocessing.harmonize import harmonize_index
from ...data_models.ProjectSettings import ProjectSettings
import pandas as pd


def create_samples(
    dataset_name: str = "jackson",
    debug: bool = False,
    raw_dir: Path = None,
    processed_dir: Path = None,
    only_complete_samples: bool = True,
):
    ps = ProjectSettings(dataset_name=dataset_name)
    ds = Jackson(raw_dir=ps.raw_dir, processed_dir=ps.processed_dir)
    df_samples_path = ps.processed_dir / "samples.parquet"

    if df_samples_path.exists():
        df_samples = pd.read_parquet(df_samples_path)
        df_samples = df_samples.dropna()
    else:
        df_samples = ds.load()
        df_samples = df_samples.dropna()
        df_samples.astype(str).to_parquet(df_samples_path)

        for sample in df_samples.index:
            harmonize_index(
                mask_path=df_samples.loc[sample].mask_path,
                expr_path=df_samples.loc[sample].obs_expr_path,
                labels_path=df_samples.loc[sample].obs_labels_path,
                loc_path=df_samples.loc[sample].obs_loc_path,
                spat_path=df_samples.loc[sample].obs_spatial_path,
            )
    # spls_remove = df_samples[df_samples.isna().any(axis=1)].index

    if debug:
        df_samples = df_samples.iloc[:10]

    samples = ds.load_samples(df_samples)
    for sample in samples:
        sample.to_pickle(ps.samples_dir / f"{sample.name}.pkl")
    # if remove and len(spls_remove):
    #     for sample in spls_remove:
    #         files = processed_dir.rglob(f"{sample}*")
    #         for file in files:
    #             logging.info(f"removing {file}")
    #             file.unlink()
