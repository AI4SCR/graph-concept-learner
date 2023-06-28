# Generates concept graph dataset for every concept
if config["normalize_with"] != "None":
    path_to_so = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/so_norm.pkl"
else:
    path_to_so = f"{root}/intermediate_data/so.pkl"

rule gen_rand_concept_graph_dataset:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/randomized_data/filtered_sample_ids_and_labels.csv",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/dataset_configs/{{concept}}_radius.yaml",
        path_to_so
    output:
        directory(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/randomized_data/{{concept}}_radius/")
    resources:
        cores=2,
        mem="6G",
        queue="x86_24h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/gen_rand_concept_graph_dataset/{{concept}}_radius"
    shell:
        "4_generate_graph_datasets/scripts/generate_concept_graph_dataset.py {input} {output}"
