# Train a GNN model specifies by a config, log results and save weights.
rule pretrain_concept:
    input:
        #f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/pretrain_model_configs", # Dependance on configs generation
        #get_all_graphs_for_a_concept, # Dependence on all graphs.
        cfg=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/pretrain_model_configs/{{config_id}}.yaml",
        splits=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/CV_folds/folds/{{fold}}.csv",
        concept=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/attributed/{{concept}}/{{fold}}"
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/{{concept}}/{{fold}}/{{config_id}}/test_conf_mat_from_best_val_balanced_accuracy.png",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/{{concept}}/{{fold}}/{{config_id}}/test_conf_mat_from_best_val_weighted_f1_score.png",
        out_files=expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/"+"{{concept}}/{{fold}}/{{config_id}}/best_val_{metric_name}.pt", metric_name=config["follow_this_metrics"]),
    params:
        normalized_with=normalized_with,
        split_strategy=split_how,
        run_type="pretrain_concept",
        randomize="False",
        mlflow_on_remote_server=mlflow_on_remote_server,
        mlflow_uri=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/mlruns"
    resources:
        cores="1+1",
        mem="3G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/pretrain_concept/{{concept}}/{{fold}}/{{config_id}}"
    shell:
        "source scripts/setup_MLflow.sh {params.mlflow_on_remote_server} && "
        "6_pretrain/scripts/pretrain_concept.py "
        "{input.cfg} {input.splits} {input.concept} "
        "{params.normalized_with} {params.split_strategy} {params.run_type} {params.randomize} "
        "{params.mlflow_on_remote_server} {params.mlflow_uri} "
        "{config[prediction_target]} {config[root]} {config[log_frequency]} "
        "{output.out_files}"
