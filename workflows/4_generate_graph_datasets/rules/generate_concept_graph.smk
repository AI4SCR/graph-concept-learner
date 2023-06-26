# Generates concept graph for every concept and sample in fliter samples
rule generate_concept_graph:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/dataset_configs/{{concept}}_contact.yaml",
        f"{root}/intermediate_data/so.pkl",
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/unattributed/{{concept}}_contact/{{spl_id}}.pt"
    params:
        spl_id = "{spl_id}"
    resources:
        cores=1,
        mem="2G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/generate_concept_graph/{{concept}}_contact/{{spl_id}}"
    shell:
        "4_generate_graph_datasets/scripts/generate_concept_graph.py {params.spl_id} {input} {output}"
