#!/usr/bin/env python3
import pandas as pd
import sys

# Unpack inputs
program, md_path_b, md_path_z, path_to_output = sys.argv
# md_path_b = snakemake.input[0]
# md_path_z = snakemake.input[1]
# path_to_output = snakemake.output[0]
# md_path_z = "/dccstor/cpath_data/datasets/GCL/ER/raw_data/unzipped/Data_publication/ZurichTMA/Zuri_PatientMetadata.csv"
# md_path_b = "/dccstor/cpath_data/datasets/GCL/ER/raw_data/unzipped/Data_publication/BaselTMA/Basel_PatientMetadata.csv"
# path_to_output = "/dccstor/cpath_data/datasets/GCL/ER/int_data/spl_meta_data.csv"

# Read data
md_z = pd.read_csv(md_path_z)
md_b = pd.read_csv(md_path_b)

# Add cohort column
md_z["cohort"] = "zurich"
md_b["cohort"] = "basel"

# Get colum interections
z = set(md_z.columns.values)
b = set(md_b.columns.values)
intx = z.intersection(b)

# Concatenate
md = pd.concat([md_z[list(intx)], md_b[list(intx)]])

# Save to file
md.to_csv(path_to_output, index=False)

print("Done!")
