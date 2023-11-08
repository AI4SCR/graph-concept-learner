# Generates concept graph for every concept and sample in fliter samples
if config["normalize_with"] != "None":
    path_to_so = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/so_norm.pkl"
else:
    path_to_so = f"{root}/intermediate_data/so.pkl"

rule gen_rand_concept_graph:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/randomized_data/filtered_sample_ids_and_labels.csv",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/concept_configs/{{concept}}_contact.yaml",
        path_to_so,
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/randomized_data/{{concept}}_contact/{{spl_id}}.pt"
    params:
        spl_id = "{spl_id}"
    resources:
        cores=1,
        mem="2G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/gen_rand_concept_graph/{{concept}}_contact/{{spl_id}}"
    shell:
        "4_generate_graph_datasets/scripts/generate_concept_graph.py {params.spl_id} {input} {output}"
