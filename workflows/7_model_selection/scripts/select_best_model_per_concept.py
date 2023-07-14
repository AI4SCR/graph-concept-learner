#!/usr/bin/env python3

import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ruamel import yaml
import sys

### Debugging input ###
# concept="tumor_contact_ER"
# normalized_with="normalized_with_min_max"
# metric_name="balanced_accuracy"
# prediction_target="ERStatus"
# dataset_name="jackson"

### Set up ###
(
    program_name,
    concept,
    normalized_with,
    metric_name,
    prediction_target,
    dataset_name,
    output_figure,
    output_config,
) = sys.argv

# Define experiment name
experiment_name = f"san_{dataset_name}_{prediction_target}"

# Defin query
query = f"params.run_type = 'pretrain_concept' and params.concept = '{concept}' and params.folder_name = '{normalized_with}'"

# Load data
df = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string=query,
)

# Define group by columns
df["cfg_id"] = (
    df["params.pool"]
    + df["params.gnn"]
    + df["params.lr"].astype(str)
    + df["params.jk"]
    + df["params.norm"]
    + df["params.num_layers"].astype(str)
    + df["params.scaler"].astype(str)
    + df["params.num_layers_MLP"].astype(str)
    + df["params.act"]
    + df["params.batch_size"].astype(str)
    + df["params.optim"]
    + df["params.n_epoch"].astype(str)
    + df["params.scheduler"]
)

### Choose best performing model ###
# Compute statistics
stats = (
    df[["cfg_id", f"metrics.best_val_{metric_name}"]].groupby(by="cfg_id").describe()
)

# Select row with highest median
best_params = stats.loc[stats["50%"].idxmax()]

# Select checkpoint
runs_best_config = df.loc[(df["cfg_id"] == best_params.cfg_id)]
best_run = runs_best_config.loc[
    runs_best_config[f"metrics.best_val_{metric_name}"].idmax()
]

# Initialize output dictionary
checkpoint_and_config_per_concept = {}
checkpoint_and_config_per_concept[concept] = {
    "checkpoint": os.path.join(
        best_run.path_output_models, f"best_val_{metric_name}.pt"
    ),
    "config": best_run.path_input_config,
}

# Save config to file
# Write config
with open(output_config, "w") as file:
    yaml.dump(checkpoint_and_config_per_concept, file, Dumper=yaml.RoundTripDumper)

### Make figure and save to file ###
# Seaborn theme to "white" and modifies the "axes.facecolor" parameter to have a transparent background.
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Initializing the FacetGrid object
# Color palette
pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
g = sns.FacetGrid(
    df[["cfg_id", f"metrics.best_val_{metric_name}"]],
    row="cfg_id",
    hue="cfg_id",
    aspect=15,
    height=0.5,
    palette=pal,
)

# Adding a reference line
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Adjusting the subplots and removing details
g.figure.subplots_adjust(hspace=-0.25)
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
plt.savefig(output_figure, bbox_inches="tight", dpi=300)
