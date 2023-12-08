rule select_best_model_per_concept:
    input:
        expand(
            f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/"+"{{labels_permuted}}/{{attribute_config}}/{{concept}}/{cfg_id}/{fold}/{seed}/best_val_{metric_name}.pt",
            metric_name=config["follow_this_metrics"],
            seed=[f"seed_{i}" for i in range(config["n_seeds"])],
            fold=[f"fold_{i}" for i in range(config["n_folds"])],
            cfg_id=[os.path.splitext(f)[0] for f in os.listdir(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/pretrain_model_configs") if f.endswith(".yaml")]
        ),
        mlflow_uri=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/mlruns/flag.txt"
    output:
        cfg=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_model_per_concept/{{labels_permuted}}/{{attribute_config}}/{{concept}}.yaml",
        run_id=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/pretrain_results/mlflow_run_ids/{{labels_permuted}}/{{attribute_config}}/{{concept}}.txt",
        plot=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/pretrain_results/model_performance_distribution_per_concept/{{labels_permuted}}/{{attribute_config}}/{{concept}}.png",
    params:
        dataset_name=dataset_name,
        prediction_target=prediction_target,
        split_strategy=split_how,
        normalized_with=normalized_with,
        concept="{concept}",
        attribute_config="{attribute_config}",
        labels_permuted="{labels_permuted}",
        run_type="pretrain_concept",
        metric_name="balanced_accuracy",
        mlflow_on_remote_server=mlflow_on_remote_server
    resources:
        cores="1",
        mem="2G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/select_best_model_per_concept/{{labels_permuted}}/{{attribute_config}}/{{concept}}"
    shell:
        "source scripts/setup_MLflow.sh {params.mlflow_on_remote_server} && "
        "7_model_selection/scripts/select_best_model_per_concept.py "
        "{params.dataset_name} "
        "{params.prediction_target} "
        "{params.split_strategy} "
        "{params.normalized_with} "
        "{params.concept} "
        "{params.attribute_config} "
        "{params.labels_permuted} "
        "{params.run_type} "
        "{params.metric_name} "
        "$(dirname {input.mlflow_uri}) "
        "{output}"
