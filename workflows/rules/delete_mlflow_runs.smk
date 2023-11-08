# Delete some mlflow run. Dont use unless you know what you are donig.
rule delete_mlflow_runs:
    input:
        path_to_all_concepts=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/concept_configs"
    params:
        folder_name=normalized_with,
        split_strategy=split_how
    resources:
        cores=1,
        mem="1g",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/delete_mlflow_runs/delete_mlflow_runs"
    shell:
        "source scripts/setup_MLflow.sh && "
        "scripts/delete_mlflow_runs.py {input.path_to_all_concepts} "
        "{config[prediction_target]} {params.folder_name} {params.split_strategy} {config[root]}"
