#!/usr/bin/env python3
import os
import sys
import mlflow

# Unpack inputs
(
    program_name,
    root,  # Path to the output file with the final model
    pred_target,
    mlflow_on_remote_server,
    mlflow_uri,
) = sys.argv

# Define mlflow experiment
if mlflow_on_remote_server == "False":
    mlflow.set_tracking_uri(mlflow_uri)
else:
    os.makedirs(mlflow_uri)
    os.path.join(mlflow_uri, "README.txt")
    with open(mlflow_uri, "w") as f:
        f.write(
            """"
            Since mlflow_on_remote_server == False in the main_config.yaml, \
            this folder should be empty \
            (with the exception of the file containing this message).
            """
        )

# Define mlflow experiment
dataset_name = os.path.basename(root)
mlflow.set_experiment(experiment_name=f"san_{dataset_name}_{pred_target}")
