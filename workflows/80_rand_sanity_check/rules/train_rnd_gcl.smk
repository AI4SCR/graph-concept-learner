rule train_rnd_gcl:
    input:
        # f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/train_model_configs", # Dependance w config gen
        paths_to_pretrain_configs=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/pretrain_results/best_model_per_concept/best_rnd_model_per_concept.yaml",
        path_to_train_config=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/train_model_configs/{{config_id}}.yaml",
        path_to_datasets=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/randomized_data",
        splits=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/randomized_data/samples_splits.csv",
    output:
        expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/random_gcl/"+"{{config_id}}/test_conf_mat_from_best_val_{metric_name}.png", metric_name=config["follow_this_metrics"]),
        out_files=expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/random_gcl"+"/{{config_id}}/best_val_{metric_name}.pt", metric_name=config["follow_this_metrics"]),
    params:
        exclude=["all_cells_radius", "all_cells_contact"], # Concept/datasets present in path_to_datasets to exclude
        folder_name=normalized_with,
        split_strategy=split_how,
        run_type="train_rnd_gcl",
    resources:
        cores="1+1",
        mem="3G",
        queue="x86_24h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/train_gcl/{{config_id}}"
    shell:
        "source scripts/setup_MLflow.sh && "
        "7_train/scripts/train_gcl.py "
        "{input.paths_to_pretrain_configs} {input.path_to_train_config} {input.path_to_datasets} {input.splits} "
        "{params.folder_name} {params.split_strategy} "
        "{config[prediction_target]} {config[root]} {config[log_frequency]} "
        "{output.out_files} "
        "{params.run_type} "
        "{params.exclude}"