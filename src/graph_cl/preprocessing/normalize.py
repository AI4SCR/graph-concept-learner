# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


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
