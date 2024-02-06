import pickle
import pandas as pd
import numpy as np

# import sys
import os
from sklearn.model_selection import StratifiedShuffleSplit
from graph_cl.configuration import CONFIG


def split_basel_leave_zurich_as_external():
    so_file = CONFIG.data.intermediate

    root = CONFIG.experiment.root
    filtered_samples = root / "filter" / "filtered_sample_ids_and_labels.csv"
    prediction_target = CONFIG.experiment.prediction_target
    train, test, val = CONFIG.split.train, CONFIG.split.test, CONFIG.split.val
    n_folds = CONFIG.split.n_folds

    out_dir_folds = root / "meta_data" / "CV_folds" / "folds"
    out_dir_folds.mkdir(parents=True, exist_ok=True)
    out_dir_prop = root / "meta_data" / "CV_folds" / "proportions"
    out_dir_prop.mkdir(parents=True, exist_ok=True)

    splits = np.array([float(train), float(test), float(val)])

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
    df_zuri = df_zuri.assign(split="test")

    # Split train and others
    sss = StratifiedShuffleSplit(
        n_splits=int(n_folds), train_size=splits[0], random_state=0
    )
    split_iter = sss.split(df_basel.index.values, df_basel[prediction_target])

    # Assign labels and write sv for every fold
    for i, fold in enumerate(split_iter):
        train_idxs, val_idxs = fold

        # Assign label and take other
        df_basel.loc[df_basel.index.values[train_idxs], "split"] = "train"
        df_basel.loc[df_basel.index.values[val_idxs], "split"] = "val"

        # Concat all
        df = pd.concat([df_basel, df_zuri])

        # Name output files and make dirs if absent
        fold_split_file = out_dir_folds / f"fold_{i}.csv"
        fold_proportions_file = out_dir_prop / f"fold_{i}.csv"

        # Write files
        df.to_csv(fold_split_file, index=True)
        df.reset_index().rename(columns={"index": "core"}).groupby(
            [prediction_target, "split"]
        )["core"].count().to_csv(fold_proportions_file, index=True)


def split_both_cohorts():
    pass


def split_zurich_leave_basel_as_external():
    pass


def split_samples():
    method = CONFIG.split.method
    if method == "split_basel_leave_zurich_as_external":
        split_basel_leave_zurich_as_external(CONFIG)
    elif method == "split_both_cohorts":
        split_both_cohorts(CONFIG)
    elif method == "split_zurich_leave_basel_as_external":
        split_zurich_leave_basel_as_external(CONFIG)
    else:
        raise ValueError(f"Unknown split method: {method}")
