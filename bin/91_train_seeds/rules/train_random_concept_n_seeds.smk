# Train a GNN model specifies by a config, log results and save weights.
rule train_random_concept_n_seeds:
    input:
        #f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/pretrain_model_configs", # Dependance on configs generation
        #get_all_graphs_for_a_concept, # Dependence on all graphs.
        cfg=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_{{concept}}/{{config_id}}.yaml",
        splits=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/samples_splits.csv",
        concept=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/processed_data/{{concept}}"
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/{{concept}}/{{config_id}}/test_conf_mat_from_best_val_balanced_accuracy.png",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/{{concept}}/{{config_id}}/test_conf_mat_from_best_val_weighted_f1_score.png",
        out_files=expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/"+"{{concept}}/{{config_id}}/best_val_{metric_name}.pt", metric_name=config["follow_this_metrics"]),
    params:
        folder_name=normalized_with,
        split_strategy=split_how,
        run_type="pretrain_rnd_concept",
        randomize="True",
    resources:
        cores="1+1",
        mem="3G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/pretrain_concept/randomized_data/{{concept}}/{{config_id}}"
    shell:
        "source scripts/setup_MLflow.sh && "
        "6_pretrain/scripts/pretrain_concept.py "
        "{input.cfg} {input.splits} {input.concept} "
        "{params.folder_name} {params.split_strategy} {params.run_type} {params.randomize} "
        "{config[prediction_target]} {config[root]} {config[log_frequency]} "
        "{output.out_files}"
