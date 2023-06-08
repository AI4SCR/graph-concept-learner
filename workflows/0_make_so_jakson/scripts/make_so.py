#!/usr/bin/env python3

import pickle
import sys
import pandas as pd
import numpy as np
import os.path as osp
from spatialOmics import SpatialOmics

### Read in inputs and output paths ###
(prog_name, uns_path, spl_path, obs_dir, X_dir, masks_dir, out_file) = sys.argv

uns_df = pd.read_csv(uns_path, skiprows=range(1, 10, 1))
spl_df = pd.read_csv(spl_path, index_col="core")

# obs_dir = snakemake.input[2]
# X_dir = snakemake.input[3]
# masks_dir = snakemake.input[4]
# out_file = snakemake.output[0]

# uns_df = pd.read_csv(
#     "/Users/ast/Documents/GitHub/datasets/ER/raw_data/unzipped/Data_publication/Basel_Zuri_StainingPanel.csv",
#     skiprows=range(1, 10, 1),
# )
# spl_df = pd.read_csv(
#     "/Users/ast/Documents/GitHub/datasets/ER/int_data/spl_meta_data.csv",
#     index_col="core",
# )
# obs_dir = "/Users/ast/Documents/GitHub/datasets/ER/int_data/sct_data"
# X_dir = "/Users/ast/Documents/GitHub/datasets/ER/int_data/scc_data"
# masks_dir = "/Users/ast/Documents/GitHub/datasets/ER/raw_data/unzipped/OMEnMasks/Basel_Zuri_masks"
# img_dir = "/Users/ast/Documents/GitHub/datasets/ER/raw_data/unzipped/OMEnMasks/ome"
# out_file = "/Users/ast/Documents/GitHub/datasets/ER/int_data/so_w_imgs.pkl"

### Assign object level data ###
# Instantiate spatial omics object
so = SpatialOmics()

# Assign metadata to so object
so.spl = spl_df
so.uns["staining_panel"] = uns_df

# Get list of all samples.
spls = spl_df.index.values

# Get column names for obs
path_to_colnames = osp.join(obs_dir, "core.csv")
colnames_obs = pd.read_csv(path_to_colnames).columns.values

# Get column names for X
path_to_colnames = osp.join(X_dir, "core.csv")
colnames_X = pd.read_csv(path_to_colnames).columns.values

# Make map for numerica cell_type class
keys = ["Stroma", "Immune", "Vessel", "Tumor"]
numeric_labels = list(range(0, len(keys)))
map_to_numeric = dict(zip(keys, numeric_labels))

# Get names of all the channels in X
spl = so.spl.index.values[0]
path_to_X = osp.join(X_dir, f"{spl}.csv")
X_df = pd.read_csv(path_to_X, names=colnames_X).pivot(
    index="CellId", columns="channel", values="mc_counts"
)
X_df.index.names = ["cell_id"]
colnames_X_wide = X_df.columns.values

# Get metal tag labels
unique_metal_tags = np.unique(uns_df["Metal Tag"].values)

# Init dictionary
map_to_metaltag = {}

# Map matal tag label to column name
for string in colnames_X_wide:
    for substring in unique_metal_tags:
        if substring in string:
            map_to_metaltag[string] = substring


### Assign sample level data ###
for spl in spls:

    # Load obs df
    path_to_obs = osp.join(obs_dir, f"{spl}.csv")
    so.obs[spl] = pd.read_csv(
        path_to_obs, names=colnames_obs, index_col="ObjectNumber_renamed"
    ).drop(columns="core")
    so.obs[spl]["class_id"] = so.obs[spl]["Class"].map(map_to_numeric)
    so.obs[spl].index.names = ["cell_id"]

    # Reorder columns
    so.obs[spl] = so.obs[spl][
        [
            "id",
            "cluster",
            "Cell_type",
            "Class",
            "class_id",
            "PhenoGraph",
            "Location_Center_X",
            "Location_Center_Y",
        ]
    ]

    # Load X df
    path_to_X = osp.join(X_dir, f"{spl}.csv")
    so.X[spl] = pd.read_csv(path_to_X, names=colnames_X).pivot(
        index="CellId", columns="channel", values="mc_counts"
    )
    so.X[spl].index.names = ["cell_id"]

    # Join morphological features into obs
    so.obs[spl] = so.obs[spl].join(
        so.X[spl][
            [
                "Area",
                "Eccentricity",
                "EulerNumber",
                "Extent",
                "MajorAxisLength",
                "MinorAxisLength",
                "Number_Neighbors",
                "Orientation",
                "Percent_Touching",
                "Perimeter",
                "Solidity",
            ]
        ],
        how="left",
    )

    # Colnames in lowercase
    so.obs[spl].rename(str.lower, axis="columns", inplace=True)

    # Rename class as cell_class
    so.obs[spl].rename(columns={"class": "cell_class"}, inplace=True)

    # Reneame columns with metal tag
    so.X[spl] = so.X[spl].rename(columns=map_to_metaltag)

    # Get metal tags present in sample
    metaltas_in_sampl = so.X[spl].columns.intersection(unique_metal_tags)

    # Drop other columns
    so.X[spl] = so.X[spl][metaltas_in_sampl]

    # Load mask
    file_full_name = spl_df.loc[spl]["FileName_FullStack"]
    file_base_name = osp.splitext(file_full_name)[0]
    path_to_mask = osp.join(masks_dir, f"{file_base_name}_maks.tiff")
    so.add_mask(spl, "cellmasks", path_to_mask, to_store=False)

    # Load images (file becomes heavy)
    # path_to_image = osp.join(img_dir, file_full_name)
    # so.add_image(spl, path_to_image, to_store=False)

# Drop metal tags only present in either of the cohorts
metal_tags_in_all_spls = set(unique_metal_tags)
for spl in spls:
    metal_tags_in_all_spls = metal_tags_in_all_spls.intersection(so.X[spl].columns)

for spl in spls:
    # Drop columns
    so.X[spl] = so.X[spl][list(metal_tags_in_all_spls)]

### Write to file ###
with open(out_file, "wb") as f:
    pickle.dump(so, f)
