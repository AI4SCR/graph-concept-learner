# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from torch_geometric.data import Data
import torch

# %%
def normalize(
    feat: pd.DataFrame, fold_info: pd.DataFrame, output_dir: Path, method: str, **kwargs
):
    scaler = MinMaxScaler()
    for split in ["train", "val", "test"]:
        split_feat = feat.loc[fold_info[fold_info.split == split].index, :]
        assert split_feat.index.get_level_values("cell_id").isna().any() == False

        feat_norm = _normalize_features(split, split_feat, scaler=scaler, **kwargs)
        for core, grp_data in feat_norm.groupby("core"):
            grp_data.to_parquet(output_dir / f"{core}.parquet")


def _normalize_features(split, split_feat, scaler, cofactor, censoring):
    X = split_feat.values.copy()

    # arcsinh transform
    np.divide(X, cofactor, out=X)
    np.arcsinh(X, out=X)

    # censoring
    thres = np.quantile(X, censoring, axis=0)
    for idx, t in enumerate(thres):
        X[:, idx] = np.where(X[:, idx] > t, t, X[:, idx])
    if split == "train":
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return pd.DataFrame(X, index=split_feat.index, columns=split_feat.columns)


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
