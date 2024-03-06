#!/usr/bin/env python3
import pickle
import pandas as pd
import numpy as np
import sys

# Read input output paths
(
    prog_name,
    so_file,
    cell_type_csv_path,
    filered_samples,
    path_to_count,
    path_to_count_HRless,
) = sys.argv

# Load so obj
with open(so_file, "rb") as f:
    so = pickle.load(f)

# Load filtered samples
filters_samples_csv = pd.read_csv(filered_samples)
y = filters_samples_csv["ERStatus"].values
spls = filters_samples_csv["core"].values

# Get array of cell types from the dataset.
cell_type_csv = pd.read_csv(cell_type_csv_path, delimiter=";")
cell_type_csv.rename(str.lower, axis="columns", inplace=True)
cell_type_csv.columns = cell_type_csv.columns.str.replace(" ", "_")
cell_type_csv["cell_type"] = cell_type_csv["cell_type"].str.replace(" ", "_")
cell_types = cell_type_csv["cell_type"].values
dict(zip(cell_types, [0] * len(cell_types))).keys

# Count cell types per sample
X = np.empty([0, len(set(cell_types))], dtype=int)
for spl in spls:
    new_vec = dict(zip(cell_types, [0] * len(cell_types)))
    cell_counts = (
        so.obs[spl].groupby(["cell_type"], dropna=True).count()["id"].to_dict()
    )
    new_vec.update(cell_counts)
    new_vec = np.array(list(new_vec.values()))
    X = np.vstack([X, new_vec])

# Get rid of empty samples, or samples with not cell type info
# list_of_indexes = []
# list_of_samples = []
# for i in range(X.shape[0]):
#     if X[i].sum() == 0:
#         list_of_indexes.append(i)
#         list_of_samples.append(spls[i])

# X = np.delete(X, list_of_indexes, 0)

# Normalize X
X_norm = (X.T / X.sum(axis=1)).T

# Make into a data frames and save to file.
df = pd.DataFrame(
    X, columns=dict(zip(cell_types, [0] * len(cell_types))).keys(), index=spls
)
df["y"] = y

df_norm = pd.DataFrame(
    X_norm, columns=dict(zip(cell_types, [0] * len(cell_types))).keys(), index=spls
)
df_norm["y"] = y

# Make the HR less files
for cell_type in cell_types:
    if "HR" in cell_type:
        cell_types = cell_types[cell_types != cell_type]
    elif "Ecadh" in cell_type:
        cell_types = cell_types[cell_types != cell_type]

# Write to output
cell_types = np.append(cell_types, "y")
df.to_csv(path_to_count)
df[cell_types].to_csv(path_to_count_HRless)
