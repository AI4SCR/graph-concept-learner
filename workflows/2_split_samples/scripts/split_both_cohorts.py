#!/usr/bin/env python3

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

# so_file = "/Users/ast/Documents/GitHub/datasets/jakson/int_data/so.pkl"
# spls_file = "/Users/ast/Documents/GitHub/datasets/jakson/prediction_targets/ERStatus/meta_data/filtered_sample_ids_and_labels.csv"
# prediction_target = "ERStatus"
splits = np.array([float(train), float(test), float(val)])
split_names = ["train", "test", "val"]

# Load so object
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Load list of filtered samples
# expects column named "core"
spls = pd.read_csv(filtered_samples)["core"].values

# Filter out irrelevant metadata
relevant_metadata = so.spl.filter(items=spls, axis=0)

# Make cohort specific PID
# expects column named "PID" (int) and "cohort" (str)
relevant_metadata["PID_cohort"] = (
    relevant_metadata["PID"].astype(str) + "_" + relevant_metadata["cohort"]
)
assert all(
    relevant_metadata.groupby(["PID_cohort"])[prediction_target].nunique() == 1
), "Not all samples assigned to one patinet have the same prediction label."

# Count samples per class
class_count = np.array([])

for cls in relevant_metadata[prediction_target].unique():
    class_count = np.append(
        class_count, (relevant_metadata[prediction_target] == cls).sum()
    )

# Get number of samples per class and split
n_spls_pc_ps = class_count.reshape((len(class_count), 1)) * splits

# Create new column indicating split
relevant_metadata["split"] = "not_considered"

# Set seed
np.random.seed(123)

# Loop over classes
for i, cls in enumerate(relevant_metadata[prediction_target].unique()):

    # All patients classified as cls
    bag = np.unique(
        relevant_metadata.loc[relevant_metadata[prediction_target] == cls][
            "PID_cohort"
        ].values
    )

    # Loop over splits
    for j, split in enumerate(split_names):

        # If on last split just assign all remaining samples to the validation split
        if split == "val":
            idx = relevant_metadata.query(f"PID_cohort in {list(bag)}").index
            relevant_metadata.loc[idx, "split"] = "val"
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
            pid_spls = relevant_metadata.loc[
                relevant_metadata["PID_cohort"] == pid
            ].index.values

            # Increase count
            count = count + len(pid_spls)

            # Annotate those samples as belonging to split
            relevant_metadata.loc[
                relevant_metadata["PID_cohort"] == pid, "split"
            ] = split

# Save list of samples and their asocciated prediction labels.
relevant_metadata[["split"]].to_csv(out_file, index=True)
relevant_metadata.reset_index().rename(columns={"index": "core"}).groupby(
    [prediction_target, "split"]
)["core"].count().to_csv(metda_data)
