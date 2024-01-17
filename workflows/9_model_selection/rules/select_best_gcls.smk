def get_gcl_checkpoints(wildcards):
    return expand(
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/gcls/"+"{{attribute_config}}/{cfg_ids_concept_sets}/{cfg_ids_models}/{{labels_permuted}}/{fold}/{seed}/best_val_{metric_name}.pt",
        metric_name=config["follow_this_metrics"],
        seed=[f"seed_{i}" for i in range(config["n_seeds"])],
        fold=[f"fold_{i}" for i in range(config["n_folds"])],
        cfg_ids_concept_sets=get_concept_sets_ids(wildcards),
        cfg_ids_models=[os.path.splitext(f)[0] for f in os.listdir(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/train_configs/models") if f.endswith(".yaml")],
    )

def get_concept_sets_ids(wildcards):
    p = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/train_configs/concept_sets/{wildcards.attribute_config}/{wildcards.labels_permuted}"
    return [os.path.splitext(f)[0] for f in os.listdir(p) if f.endswith(".yaml")]

rule select_best_gcls:
    input:
        get_gcl_checkpoints,
        mlflow_uri=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/mlruns/flag.txt"
    output:
        output_cfg_id=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/train_results/{{attribute_config}}/{{labels_permuted}}/cfg_id.yaml",
        run_id=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/train_results/{{attribute_config}}/{{labels_permuted}}/mlflow_run_ids.txt",
        plot=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/train_results/{{attribute_config}}/{{labels_permuted}}/model_performance_distributions.png",
    params:
        dataset_name=dataset_name,
        prediction_target=prediction_target,
        split_strategy=split_how,
        normalized_with=normalized_with,
        attribute_config="{attribute_config}",
        labels_permuted="{labels_permuted}",
        run_type="train_gcl",
        metric_name="balanced_accuracy",
        mlflow_on_remote_server=mlflow_on_remote_server
    resources:
        cores="1",
        mem="2G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/select_best_gcls/{{attribute_config}}/{{labels_permuted}}"
    shell:
        "source scripts/setup_MLflow.sh {params.mlflow_on_remote_server} && "
        "9_model_selection/scripts/select_best_gcls.py "
        "{params.dataset_name} "
        "{params.prediction_target} "
        "{params.split_strategy} "
        "{params.normalized_with} "
        "{params.attribute_config} "
        "{params.labels_permuted} "
        "{params.run_type} "
        "{params.metric_name} "
        "$(dirname {input.mlflow_uri}) "
        "{output}"
