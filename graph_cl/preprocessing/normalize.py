# %%
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from graph_cl.configuration import load_config
from spatialOmics import SpatialOmics


# %%


def min_max(so, df_fold: pd.DataFrame, cofactor: int, censoring: float):
    so_norm = SpatialOmics()
    so_norm.spl = so.spl.loc[df_fold.index]

    for grp_name, grp_data in df_fold.groupby("split"):
        # gather all X across samples in the group
        grp_x = pd.concat(
            [
                so.X[core].assign(core=core).set_index("core", append=True)
                for core in grp_data.index
            ]
        )

        # arcsinh transform
        np.divide(grp_x, cofactor, out=grp_x)
        np.arcsinh(grp_x, out=grp_x)

        # censoring
        for col_name in grp_x.columns:
            thres = grp_x[col_name].quantile(censoring)
            grp_x.loc[grp_x[col_name] > thres, col_name] = thres

        # min-max normalization
        minMax = MinMaxScaler()
        grp_x = pd.DataFrame(
            minMax.fit_transform(grp_x), index=grp_x.index, columns=grp_x.columns
        )

        for core in grp_data.index:
            so_norm.obs[core] = so.obs[core]
            so_norm.X[core] = grp_x.loc[(slice(None), core), :].droplevel("core")
            so_norm.masks[core] = so.masks[core]

    return so_norm


def normalize_fold(
    so_path: Path, fold_path: Path, output_dir: Path, config_path: Path, **kwargs
):
    config = load_config(config_path)
    method = config.data.processing.normalize.method

    with open(so_path, "rb") as f:
        so = pickle.load(f)

    df_fold = pd.read_csv(fold_path, index_col="core")

    if method == "min_max":
        cofactor = config.data.processing.normalize.cofactor
        censoring = config.data.processing.normalize.censoring
        so_norm = min_max(
            so=so,
            df_fold=df_fold,
            cofactor=cofactor,
            censoring=censoring,
        )
    else:
        raise NotImplementedError()

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{fold_path.stem}.pkl", "wb") as f:
        pickle.dump(so_norm, f)
