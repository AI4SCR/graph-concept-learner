# Checks against mlflow if all concept configs combinations have been pretrained.
rule parse_train_mlflow_runs:
    input:
        path_to_all_cfgs=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/train_model_configs",
    output:
        path_dupl=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/parse_train_mlflow_runs/duplicated_runs_config_ids.txt",
        path_miss=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/parse_train_mlflow_runs/missing_runs_config_ids.txt",
        path_unf=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/parse_train_mlflow_runs/unfinished_runs_config_ids.txt"
    params:
        metric_name = config["follow_this_metrics"][1], # weighted_f1_score
        folder_name = normalized_with,
        split_strategy = split_how
    resources:
        cores=1,
        mem="1g",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/parse_train_mlflow_runs/parse_train_mlflow_runs"
    shell:
        "source scripts/setup_MLflow.sh && "
        "scripts/parse_train_mlflow_runs.py "
        "{input.path_to_all_cfgs} "
        "{config[prediction_target]} {params.folder_name} {params.split_strategy} {config[root]} "
        "{params.metric_name} "
        "{output.path_dupl} {output.path_miss} {output.path_unf}"
