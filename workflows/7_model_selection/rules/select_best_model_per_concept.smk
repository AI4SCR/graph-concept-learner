rule select_best_model_per_concept:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/{{concept}}/*/test_conf_mat_from_best_val_balanced_accuracy.png"
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_model_per_concept/{{concept}}.yaml",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/figures/models_performance_distribution_per_concept/{{concept}}.png",
    params:
        folder_name=normalized_with,
        metric_name="balanced_accuracy",
        concept_path = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/attributed/{{concept}}"
    resources:
        cores="1",
        mem="2G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/select_best_model_per_concept/{{concept}}"
    shell:
        "source scripts/setup_MLflow.sh && "
        "7_model_selection/scripts/select_best_model_per_concept.py "
        "$(basename {params.concept_path[0]}) "
        "{params.folder_name} {params.metric_name} "
        "{config[prediction_target]} $(basename {config[root]}) "
        "{output}"
