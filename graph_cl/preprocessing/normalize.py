# %%
import pickle
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from ..configuration import CONFIG

# %%


def min_max(splits_df):

    # (
    # prog_name,
    # so_file,
    # splits_df,
    # cofactor,
    # out_file,
    # ) = sys.argv

    so_file = CONFIG.data.intermediate
    cofactor = CONFIG.normalize.cofactor

    root = CONFIG.experiment.root
    out_file = root / "meta_data" / "normalized_data" / f"{splits_df.stem}.pkl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load so object
    with open(so_file, "rb") as f:
        so = pickle.load(f)

    # Get spl ids for each split
    splits = []
    split_map = pd.read_csv(splits_df, index_col="core")

    for split in split_map["split"].unique():
        ids = split_map[split_map["split"] == split].index.values
        splits.append(ids)

    # %% For every split normalize samples separately
    # for split in tqdm(splits):
    for split in splits:
        # Create empty df to agregate al data
        all_Xs = pd.DataFrame()

        # Creates a second index layer specifying the sample id and concatenates all df toghether
        for spl in split:
            all_Xs = pd.concat(
                [all_Xs, pd.concat([so.X[spl]], keys=[spl], names=["core"])]
            )

        ### Normalize ###
        # arcsinh transform
        np.divide(all_Xs, int(cofactor), out=all_Xs)
        np.arcsinh(all_Xs, out=all_Xs)

        # censoring
        for col_name in all_Xs.columns:
            thres = all_Xs[col_name].quantile(0.999)
            all_Xs.loc[all_Xs[col_name] > thres, col_name] = thres

        # min-max normalization
        minMax = MinMaxScaler()
        all_Xs[:] = minMax.fit_transform(all_Xs)

        ### Assign back the normalized Xs in so object ###
        for spl in split:
            so.X[spl] = all_Xs.loc[spl]

    # Write to output
    with open(out_file, "wb") as f:
        pickle.dump(so, f)


def standard(splits_df):
    pass


def normalize_folds():
    method = CONFIG.normalize.method
    root = CONFIG.experiment.root

    if method == "min_max":
        func = min_max
    elif method == "standard":
        func = standard

    for fold in (root / "meta_data" / "CV_folds" / "folds").glob("*.csv"):
        func(fold)
