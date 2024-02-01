# Fetch from mlflow a specified runs, make a summary and also write the paths to the models ina file for training.
rule get_best_pretrain_models:
    input:
        #get_paths_to_pretrained_models,
        concept_run_ids=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/pretrain_results/best_model_per_concept/mlflow_run_ids_concepts_{{er_status}}.txt",
        baselines_run_ids=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/pretrain_results/best_model_per_concept/mlflow_run_ids_baselines_{{er_status}}.txt",
    output:
        path_to_best_models=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/pretrain_results/best_model_per_concept/best_model_per_concept_{{er_status}}.yaml",
        path_to_summary=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/pretrain_results/best_model_per_concept/best_model_per_concept_{{er_status}}.csv",
    resources:
        cores="1",
        mem="1G",
        queue="x86_1h",
    params:
        metric_name = config["follow_this_metrics"][0], # balanced accuracy
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/get_best_pretrain_models/{{er_status}}/get_best_pretrain_models"
    shell:
        "source scripts/setup_MLflow.sh && "
        "7_model_selection/scripts/get_best_pretrain_models.py "
        "{params.metric_name} "
        "{output.path_to_best_models} {output.path_to_summary} "
        "{input.concept_run_ids} {input.baselines_run_ids}"
