#!/usr/bin/env python3
# %%
import pickle
import pandas as pd
import numpy as np
import sys

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
# splits = np.array([0.7, 0.15, 0.15])

# Load so object
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Get filteres samples list with labels and cohort info
filtered_samples_df = pd.read_csv(filtered_samples, index_col="core")
spls = filtered_samples_df.index.values
filtered_samples_df = filtered_samples_df.join(
    so.spl.filter(items=spls, axis=0)["cohort"]
)

# Set the zuirch cohort aside
df_zuri = filtered_samples_df[filtered_samples_df["cohort"] == "zurich"]
df_zuri = df_zuri.join(so.spl.loc[df_zuri.index.values, "PID"])
df_zuri["split"] = "to_be_determined"

# Get nuique PID of patients
pids = np.unique(df_zuri["PID"].values)
n_cores = df_zuri.shape[0]

# Count samples per class
class_count = np.array([])

for cls in df_zuri[prediction_target].unique():
    class_count = np.append(class_count, (df_zuri[prediction_target] == cls).sum())

# Get number of samples per class and split
n_spls_pc_ps = class_count.reshape((len(class_count), 1)) * splits

# Set seed
np.random.seed(123)
split_names = ["train", "test", "val"]

while df_zuri.groupby(["split", "ERStatus"]).count().shape[0] < 6:

    #  Loop over classes
    for i, cls in enumerate(df_zuri[prediction_target].unique()):

        # All patients classified as cls
        bag = np.unique(df_zuri.loc[df_zuri[prediction_target] == cls]["PID"].values)

        # Loop over splits
        for j, split in enumerate(split_names):

            # If on last split just assign all remaining samples to the validation split
            if split == "val":
                idx = df_zuri.query(f"PID in {list(bag)}").index
                df_zuri.loc[idx, "split"] = "val"
                continue

            # Initialize count
            count = 0
            max_spls = n_spls_pc_ps[i][j]

            # While the count is lower than the number of expected samples
            # in a given class-split, assign samples to splits
            while count < max_spls:
                # Sample one patient from bag
                pid = np.random.choice(bag)

                # Remove pid from bag
                bag = bag[bag != pid]

                # Get number of samples associated to that patient
                pid_spls = df_zuri.loc[df_zuri["PID"] == pid].index.values

                # Increase count
                count = count + len(pid_spls)

                # Annotate those samples as belonging to split
                df_zuri.loc[df_zuri["PID"] == pid, "split"] = split

# %% Fill basel
df_basel = filtered_samples_df[filtered_samples_df["cohort"] == "basel"]
df_basel = df_basel.join(so.spl.loc[df_basel.index.values, "PID"])
df_basel["split"] = "external_test"

# Join
df = pd.concat([df_zuri, df_basel])

# Save list of samples and their asocciated prediction labels.
df[[prediction_target, "split"]].to_csv(out_file, index=True)
df.groupby(["split", "ERStatus"]).count()["PID"].to_csv(metda_data)
