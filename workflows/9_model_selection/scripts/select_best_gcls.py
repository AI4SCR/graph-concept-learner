#!/usr/bin/env python3
import mlflow
import sys
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Unpack arguments
(
    program_name,
    dataset_name,
    pred_target,
    split_strategy,
    normalized_with,
    attribute_config,
    labels_permuted,
    run_type,
    metric_name,
    mlflow_uri,
    output_cfg_id,
    output_mlflow_run_id,
    output_plot,
) = sys.argv

# Set mlflow uri
mlflow.set_tracking_uri(mlflow_uri)

# Expand the metric name
metric = f"metrics.best_val_{metric_name}"

# Define query for MLFlow
query = f"""\
    params.run_type = "{run_type}" and \
    params.split_strategy = "{split_strategy}" and \
    params.attribute_config = "{attribute_config}" and \
    params.normalized_with = "{normalized_with}" and \
    params.labels_permuted = "{labels_permuted}"
    """

# Query
experiment_name = f"san_{dataset_name}_{pred_target}"
df = mlflow.search_runs(experiment_names=[experiment_name], filter_string=query)

# Compute medians for al combinations of cfg_id_concept_set & cfg_id_model
median_metrics = (
    df.groupby(["params.concept_set", "params.cfg_id"])[metric].median().reset_index()
)

# Get group with the best median
best_params = median_metrics.loc[median_metrics[metric].idxmax()]
median = best_params[metric]

# Use the values from the Series to create a DataFrame for merging
best_params_df = pd.DataFrame([best_params.values], columns=best_params.index)

# Merge with the original DataFrame to locate all runs with the best parameters
best_runs = pd.merge(
    df,
    best_params_df,
    on="params.cfg_id",
    how="inner",
    suffixes=(None, "_from_grouped"),
)

output = {
    "cfg_id": best_runs["params.cfg_id"].unique()[0],
}

# Write to file (YAML)
with open(output_cfg_id, "w") as file:
    yaml.dump(output, file, default_flow_style=False)

# Save tuple of run_ids so that the losses can be visualize in mlflow
mlflow_run_id = best_runs.loc[(best_runs[metric] - median).abs().idxmin(), "run_id"]

# Write to file (plain text file)
with open(output_mlflow_run_id, "w") as file:
    file.write(mlflow_run_id)

# Plot distributions of test and training
# Theme
metric2 = f"metrics.test_best_val_{metric_name}_{metric_name}"
df2 = df[["params.concept_set", "params.cfg_id", metric, metric2]]
sns.set_theme(
    style="whitegrid", rc={"axes.facecolor": (0, 0, 0, 0), "axes.linewidth": 1}
)

# Calculate median for each row based on the first metric
median_order = (
    df2.groupby(["params.concept_set", "params.cfg_id"])[metric]
    .median()
    .sort_values()
    .index
)

# Create a grid with a row for each configuration, ordered by the median of the first metric
g = sns.FacetGrid(
    df2,
    row="params.cfg_id",
    hue="params.cfg_id",
    aspect=9,
    height=1.6,
    xlim=(0, 1),
    row_order=median_order if len(median_order) > 1 else None,
)

# Map Kernel Density Plot for each configuration
g.map_dataframe(sns.kdeplot, x=metric, color="#377eb8", fill=True, alpha=0.7)

# Map Kernel Density Plot for the second metric
g.map_dataframe(sns.kdeplot, x=metric2, color="#ff7f00", fill=True, alpha=0.5)


# Function to draw labels
def label(x, color, label):
    ax = plt.gca()  # Get current axis
    ax.text(
        0.01,
        0.2,
        label,
        color="black",
        fontsize=10,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


# Iterate grid to plot labels
g.map(label, "params.cfg_id")

# Adjust subplots to create overlap
g.fig.subplots_adjust(hspace=-0.5)

# Remove subplot titles
g.set_titles("")

# Remove y-axis ticks and label, set x-axis label
g.set(yticks=[], ylabel="", xlabel="Balanced Accuracy")

# Remove left spine
g.despine(left=True)

# Set title
plt.suptitle(
    "GCL performance on the validation and test set for each configuration", y=0.98
)

plt.figtext(
    0.86,
    0.88,
    f"Attr cfg: {attribute_config}",
    ha="center",
    va="center",
    fontsize=12,
    color="gray",
)
plt.figtext(
    0.9,
    0.75,
    f"Labels Permuted = {labels_permuted}",
    ha="center",
    va="center",
    fontsize=12,
    color="gray",
)

# Customize the legend
legend = plt.legend(title="", loc="upper left")
legend.set_bbox_to_anchor((0.00, 1.6))
legend.get_frame().set_facecolor("#ffffff")

# Set custom labels
legend_labels = ["Validation", "Test"]
for text, label in zip(legend.get_texts(), legend_labels):
    text.set_text(label)

# Write to file (plot)
# Save the plot to a file (e.g., PNG)
plt.savefig(output_plot, bbox_inches="tight")
