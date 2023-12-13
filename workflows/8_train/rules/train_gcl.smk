# Pretrain a graph concept learner
rule train_gcl:
    input:
        mlflow_uri=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/mlruns/flag.txt",
        paths_to_concept_set=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/train_configs/concept_sets/{{attribute_config}}/{{labels_permuted}}/{{cfg_id_concept_set}}.yaml",
        path_to_train_config=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/train_configs/models/{{cfg_id_model}}.yaml",
        splits=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/CV_folds/folds/{{fold}}.csv",
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/gcls/{{attribute_config}}/{{cfg_id_concept_set}}/{{cfg_id_model}}/{{labels_permuted}}/{{fold}}/{{seed}}/test_conf_mat_from_best_val_balanced_accuracy.png",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/gcls/{{attribute_config}}/{{cfg_id_concept_set}}/{{cfg_id_model}}/{{labels_permuted}}/{{fold}}/{{seed}}/test_conf_mat_from_best_val_weighted_f1_score.png",
        out_files=expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/gcls/"+"{{attribute_config}}/{{cfg_id_concept_set}}/{{cfg_id_model}}/{{labels_permuted}}/{{fold}}/{{seed}}/best_val_{metric_name}.pt", metric_name=config["follow_this_metrics"]),
    params:
        seed="{seed}",
        mlflow_on_remote_server=mlflow_on_remote_server,
        normalized_with=normalized_with,
        split_strategy=split_how,
        run_type="train_gcl",
        labels_permuted="{labels_permuted}",

    resources:
        cores="1+1",
        mem="3G",
        queue="x86_6h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/train_gcl/{{attribute_config}}/{{cfg_id_concept_set}}/{{cfg_id_model}}/{{labels_permuted}}/{{fold}}/{{seed}}"
    shell:
        "source scripts/setup_MLflow.sh {params.mlflow_on_remote_server} && "
        "8_train/scripts/train_gcl.py "
        "{input.paths_to_concept_set} {input.path_to_train_config} {input.splits} "
        "{params.mlflow_on_remote_server} $(dirname {input.mlflow_uri}) {params.run_type} "
        "{params.normalized_with} {params.split_strategy} {params.labels_permuted} {params.seed} "
        "{config[prediction_target]} {config[root]} {config[log_frequency]} "
        "{output.out_files} "
