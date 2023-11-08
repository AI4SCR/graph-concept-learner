# Generates concept graph dataset for every concept.
# Resulting graphs have no attributes but centroid location.
rule generate_concept_graph_dataset:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/filtered_sample_ids_and_labels.csv",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs/{{concept}}_{{builder_type}}.yaml",
        f"{root}/intermediate_data/so.pkl"
    output:
        directory(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/unattributed/{{concept}}_{{builder_type,knn|radius}}/")
    resources:
        cores=2,
        mem="6G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/generate_concept_graph_dataset/{{concept}}_{{builder_type}}"
    shell:
        "4_generate_graph_datasets/scripts/generate_concept_graph_dataset.py {input} {output}"
