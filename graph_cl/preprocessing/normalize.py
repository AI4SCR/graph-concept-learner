# %%
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from graph_cl.configuration import load_config
from torch_geometric.data import Data
import torch

# %%


def arcsinh(graphs: list[Data], cofactor: int):
    for g in graphs:
        X = g.x
        torch.divide(X, torch.tensor(cofactor), out=X)
        torch.arcsinh(X, out=X)


def min_max(graphs: list[Data], censoring: float):

    # censoring
    thres = X.quantile(censoring, dim=1)
    for g in graphs:
        for idx, t in enumerate(thres):
            torch.where(g.x[:, idx] > t, t, g.x[:, idx], out=g.x[:, idx])

    X = torch.cat([g.x for g in graphs], dim=0)
    # min-max normalization
    _min, _max = X.min(), X.max()


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
