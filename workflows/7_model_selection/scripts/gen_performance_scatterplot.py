#!/usr/bin/env python3
import mlflow
import sys
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
    output,
) = sys.argv

# Set mlflow uri
mlflow.set_tracking_uri(mlflow_uri)

# Expand the metric name and define experiment name
metrics = [
    f"metrics.best_val_{metric_name}",
    f"metrics.test_best_val_{metric_name}_{metric_name}",
]
experiment_name = f"san_{dataset_name}_{pred_target}"

# Define query for MLFlow
query = f"""\
    params.run_type = "{run_type}" and \
    params.split_strategy = "{split_strategy}" and \
    params.attribute_config = "{attribute_config}" and \
    params.normalized_with = "{normalized_with}" and \
    params.labels_permuted = "{labels_permuted}"
    """

# Query
df = mlflow.search_runs(experiment_names=[experiment_name], filter_string=query)

# Compute the medians for the validation and test (one per configuration)
df = df.groupby(["params.concept", "params.cfg_id"])[metrics].median().reset_index()

# Render scatterplot
sns.regplot(data=df, x=metrics[0], y=metrics[1])

# Add labels and title
plt.xlabel("Median performance on the validation set")
plt.ylabel("Median performance on the test set")

# Set x and y-axis limits from 0 to 1
plt.xlim(0, 1)
plt.ylim(0, 1)

# Adjust layout to make the plot fit nicely
plt.tight_layout()

# Save the plot to a file (e.g., PNG, PDF, SVG)
plt.savefig(output)
