rule train_space_gm_baseline:
    input:
        path_to_cfg=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/space_gm_train_configs/{{cfg_id}}.yaml",
        path_to_splits=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/samples_splits.csv",
        path_to_data=f"{root}/intermediate_data/non_spatial_baseline/{{dataset}}/composition_vectors.csv"
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/space_gm_baseline/{{dataset}}/{{cfg_id}}/test_conf_mat_from_best_val_weighted_f1_score.png",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/space_gm_baseline/{{dataset}}/{{cfg_id}}/test_conf_mat_from_best_val_balanced_accuracy.png",
        out_files=expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/space_gm_baseline/"+"{{dataset}}/{{cfg_id}}/best_val_{metric_name}.pt", metric_name=config["follow_this_metrics"]),
    params:
        log_frequency=config["log_frequency"],
        folder_name=normalized_with,
        split_strategy=split_how
    resources:
        cores="1",
        mem="5G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/train_space_gm_baseline/{{dataset}}/{{cfg_id}}"
    shell:
        "source scripts/setup_MLflow.sh && "
        "00_baselines_non_geom/scripts/train_space_gm_baseline.py "
        "{params.folder_name} {params.split_strategy} "
        "{config[prediction_target]} {config[root]} "
        "{input.path_to_cfg} {input.path_to_splits} {input.path_to_data} "
        "{params.log_frequency} "
        "{output.out_files}"
