# Split runs
rule split_run_metrics_per_concept:
    input:
        #get_paths_to_pretrained_models,
        concept_configs_path = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/concept_configs",
    output:
        directory(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/pretrain_results/all_models_per_concept")
    params:
        folder_name=normalized_with,
        split_strategy=split_how
    resources:
        cores="1",
        mem="5G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/split_run_metrics_per_concept/split_run_metrics_per_concept"
    shell:
        "source scripts/setup_MLflow.sh && "
        "7_model_selection/scripts/split_run_metrics_per_concept.py "
        "{input.concept_configs_path} "
        "{params.folder_name} {params.split_strategy} "
        "{config[prediction_target]} {config[root]} "
        "{output}"
