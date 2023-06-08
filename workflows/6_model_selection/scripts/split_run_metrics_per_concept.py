#!/usr/bin/env python3
import mlflow
import os
import sys

# Read inputs
(
    prog_name,
    path_to_all_concepts,
    folder_name,
    split_strategy,
    pred_target,
    root,
    out_dir,
) = sys.argv

# Get concept names
CONCEPT_NAMES = [
    os.path.splitext(f)[0]
    for f in os.listdir(path_to_all_concepts)
    if os.path.splitext(f)[1] == ".yaml"
]

# Define exp name
dataset_name = os.path.basename(root)
experiment_name = f"san_{dataset_name}_{pred_target}"

# Specify info to save and how to order the table
save_cols = [
    "run_id",
    # "metrics.test_best_val_weighted_f1_score_weighted_f1_score",
    # "metrics.test_best_val_weighted_f1_score_balanced_accuracy",
    "metrics.val_balanced_accuracy",
    "metrics.val_weighted_f1_score",
    "metrics.best_val_weighted_f1_score",
    "metrics.test_best_val_balanced_accuracy_weighted_f1_score",
    "metrics.test_best_val_balanced_accuracy_balanced_accuracy",
    "params.path_input_config",
    "params.path_output_models",
]

sort_by = "metrics.test_best_val_balanced_accuracy_balanced_accuracy"

# Save to file
for concept in CONCEPT_NAMES:
    # Define path to output file
    concept_out_dir = os.path.join(out_dir, concept)
    os.makedirs(concept_out_dir, exist_ok=True)
    path_to_outfile = os.path.join(concept_out_dir, "pretrain_perfromances.csv")

    # Defin query
    query = f"params.run_type = 'pretrain_concept' and params.concept = '{concept}' and params.folder_name = '{folder_name}' and params.split_strategy = '{split_strategy}'"

    # Load data
    df = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=query,
    )

    # Save to file
    df[save_cols].sort_values(by=sort_by, ascending=False).to_csv(path_to_outfile)

# Print succses message.
print(f"Tables saved to {out_dir}")
