#!/usr/bin/env python3
# %%
import pickle
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import StratifiedShuffleSplit

# Read inputs
(
    prog_name,
    so_file,
    filtered_samples,
    prediction_target,
    train,
    test,
    val,
    out_file,
    metda_data,
) = sys.argv
splits = np.array([float(train), float(test), float(val)])

# so_file="/Users/ast/Documents/GitHub/datasets/jakson/intermediate_data/so.pkl"
# filtered_samples = "/Users/ast/Documents/GitHub/datasets/jakson/prediction_tasks/ERStatus/meta_data/filtered_sample_ids_and_labels.csv"
# prediction_target = "ERStatus"
# splits = np.array([float(0.7), float(0.15), float(0.15)])


# Load so object
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Load list of filtered samples
filtered_samples_df = pd.read_csv(filtered_samples, index_col="core")
spls = filtered_samples_df.index.values
filtered_samples_df = filtered_samples_df.join(
    so.spl.filter(items=spls, axis=0)["cohort"]
)
df_basel = filtered_samples_df[filtered_samples_df["cohort"] == "basel"]

# Assing labels to zuirch
df_zuri = filtered_samples_df[filtered_samples_df["cohort"] == "zurich"]
df_zuri.loc[:, "split"] = "test"

# Split train and others
sss = StratifiedShuffleSplit(n_splits=1, train_size=splits[0], random_state=0)
split_iter = sss.split(df_basel.index.values, df_basel[prediction_target])
for fold in split_iter:
    train_idxs, val_idxs = fold

# Assign label and take other
df_basel.loc[df_basel.index.values[train_idxs], "split"] = "train"
df_basel.loc[df_basel.index.values[val_idxs], "split"] = "val"
# df_basel_train = df_basel[df_basel["split"] == "train"]
# df_basel_other = df_basel[df_basel["split"] == "val_or_test"]

# #  Split other into val and test
# val_proportion = splits[1] / (splits[1] + splits[2])
# sss = StratifiedShuffleSplit(n_splits=1, train_size=val_proportion, random_state=0)
# split_iter = sss.split(df_basel_other.index.values, df_basel_other[prediction_target])
# for fold in split_iter:
#     val_idxs, test_idxs = fold

# # Assign label
# df_basel_other.loc[df_basel_other.index.values[val_idxs], "split"] = "val"
# df_basel_other.loc[df_basel_other.index.values[test_idxs], "split"] = "test"

# Concat all
df = pd.concat([df_basel, df_zuri])
df.to_csv(out_file, index=True)
df.reset_index().rename(columns={"index": "core"}).groupby(
    [prediction_target, "split"]
)["core"].count().to_csv(metda_data, index=True)