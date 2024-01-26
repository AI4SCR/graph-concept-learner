rule gen_performance_scatterplot:
    input:
        expand(
            f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/single_concepts/"+"{{attribute_config}}/{concepts}/{cfg_id}/{{labels_permuted}}/{fold}/{seed}/best_val_{metric_name}.pt",
            metric_name=config["follow_this_metrics"],
            seed=[f"seed_{i}" for i in range(config["n_seeds"])],
            fold=[f"fold_{i}" for i in range(config["n_folds"])],
            cfg_id=[os.path.splitext(f)[0] for f in os.listdir(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/pretrain_model_configs") if f.endswith(".yaml")],
            concepts=[os.path.splitext(f)[0] for f in os.listdir(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs") if f.endswith(".yaml")]
        ),
        mlflow_uri=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/mlruns/flag.txt"
    output:
        plot=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/pretrain_results/median_performance_scatterplot/{{attribute_config}}/{{labels_permuted}}/median_performance_scatterplot.png",
    params:
        dataset_name=dataset_name,
        prediction_target=prediction_target,
        split_strategy=split_how,
        normalized_with=normalized_with,
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
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/gen_performance_scatterplot/{{attribute_config}}/{{labels_permuted}}"
    shell:
        "source scripts/setup_MLflow.sh {params.mlflow_on_remote_server} && "
        "7_model_selection/scripts/gen_performance_scatterplot.py "
        "{params.dataset_name} "
        "{params.prediction_target} "
        "{params.split_strategy} "
        "{params.normalized_with} "
        "{params.attribute_config} "
        "{params.labels_permuted} "
        "{params.run_type} "
        "{params.metric_name} "
        "$(dirname {input.mlflow_uri}) "
        "{output.plot}"
