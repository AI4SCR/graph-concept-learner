# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from torch_geometric.data import Data
import torch


class Normalizer:
    def __init__(self, cofactor: int = 5, censoring: float = 0.999):
        self.scaler = MinMaxScaler()
        self.cofactor = cofactor
        self.censoring = censoring

    def transform(self, X: pd.DataFrame):
        x = X.values.copy()

        # arcsinh transform
        np.divide(x, self.cofactor, out=x)
        np.arcsinh(x, out=x)

        # censoring
        thres = np.quantile(x, self.censoring, axis=0)
        for idx, t in enumerate(thres):
            x[:, idx] = np.where(x[:, idx] > t, t, x[:, idx])

        x = self.scaler.transform(x)

        return pd.DataFrame(x, index=X.index, columns=X.columns)

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X, y)

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X, y)
        return self.transform(X)


# def normalize_features(
#         feat: pd.DataFrame,
#         fold_info: pd.DataFrame,
#         output_dir: Path,
#         method: str, **kwargs
# ):
#     scaler = MinMaxScaler()
#     for split in ["train", "val", "test"]:
#         split_feat = feat.loc[fold_info[fold_info.split == split].index, :]
#         assert split_feat.index.get_level_values("cell_id").isna().any() == False
#
#         feat_norm = _normalize_features(split, split_feat, scaler=scaler, **kwargs)
#         for core, grp_data in feat_norm.groupby("core"):
#             grp_data.to_parquet(output_dir / f"{core}.parquet")
#
#
# def _normalize_features(split, split_feat, scaler, cofactor, censoring):
#     X = split_feat.values.copy()
#
#     # arcsinh transform
#     np.divide(X, cofactor, out=X)
#     np.arcsinh(X, out=X)
#
#     # censoring
#     thres = np.quantile(X, censoring, axis=0)
#     for idx, t in enumerate(thres):
#         X[:, idx] = np.where(X[:, idx] > t, t, X[:, idx])
#     if split == "train":
#         X = scaler.fit_transform(X)
#     else:
#         X = scaler.transform(X)
#
#     return pd.DataFrame(X, index=split_feat.index, columns=split_feat.columns)
