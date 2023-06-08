#!/usr/bin/env python3
# %% from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

(
    prog_name,
    so_file,
    splits_df,
    cofactor,
    out_file,
) = sys.argv
# so_file = "/Users/ast/Documents/GitHub/datasets/jakson/intermediate_data/so.pkl"
# out_file = "/Users/ast/Documents/GitHub/datasets/jakson/intermediate_data/so_norm.pkl"
# splits_df = "/Users/ast/Documents/GitHub/datasets/jakson/prediction_tasks/ERStatus/meta_data/samples_splits.csv"

# Load so object
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Get spl ids for each split
split_map = pd.read_csv(splits_df, index_col="core")
train_ids = split_map[split_map["split"] == "train"].index.values
val_ids = split_map[split_map["split"] == "val"].index.values
test_ids = split_map[split_map["split"] == "test"].index.values
external_test_ids = split_map[split_map["split"] == "external_test"].index.values
splits = [train_ids, val_ids, test_ids, external_test_ids]

# For every split normalize samples separatly
# for split in tqdm(splits):
for split in splits:
    # Create empty df to agregate al data
    all_Xs = pd.DataFrame()

    # Creates a second index layer specifying the sample id and concatenates all df toghether
    for spl in split:
        all_Xs = pd.concat([all_Xs, pd.concat([so.X[spl]], keys=[spl], names=["core"])])

    ### Normalize ###
    # arcsinh transform
    cofactor = 5
    np.divide(all_Xs, int(cofactor), out=all_Xs)
    np.arcsinh(all_Xs, out=all_Xs)

    # censoring
    for col_name in all_Xs.columns:
        thres = all_Xs[col_name].quantile(0.999)
        all_Xs.loc[all_Xs[col_name] > thres, col_name] = thres

    # min-max normalization
    scaler = StandardScaler()
    all_Xs[:] = scaler.fit_transform(all_Xs)

    ### Assign back the normalized Xs in so object ###
    for spl in split:
        so.X[spl] = all_Xs.loc[spl]

# %% Write to output
with open(out_file, "wb") as f:
    pickle.dump(so, f)
