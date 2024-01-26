# Train a GNN model specifies by a config, log results and save weights.
rule set_mlflow_experiment:
    output:
        mlflow_uri=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/mlruns/flag.txt"
    params:
        mlflow_on_remote_server=mlflow_on_remote_server,
    resources:
        cores="1+1",
        mem="3G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/set_mlflow_experiment/set_mlflow_experiment"
    shell:
        "6_pretrain/scripts/set_mlflow_experiment.py "
        "{config[root]} {config[prediction_target]} "
        "{params.mlflow_on_remote_server} {output.mlflow_uri}"
