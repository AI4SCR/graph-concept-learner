#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys

(
    prog_name,
    splits_and_sometimes_labels,
    only_labels,
    preiction_target,
    split_how,
    output_file_1,
    output_file_2,
) = sys.argv

# Load list of filtered samples and prediction labels
prediction_labels = pd.read_csv(splits_and_sometimes_labels, index_col=0)
labels_df = pd.read_csv(only_labels, index_col=0)

if split_how == "both_cohorts":
    prediction_labels = prediction_labels.join(labels_df)

# Permute labels
for split in prediction_labels["split"].unique():
    prediction_labels.loc[
        prediction_labels["split"] == split, preiction_target
    ] = np.random.permutation(
        prediction_labels.loc[prediction_labels["split"] == split, preiction_target]
    )

# Wirte to output
prediction_labels.to_csv(output_file_1, index=True)
prediction_labels[preiction_target].to_csv(output_file_2, index=True)
