# Attribute every graph in a concept attirbution config cobimation
# for every fold specified in the CV.
rule attribute_concept:
    input:
        fold = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/normalized_data/{{fold}}.pkl",
        concept_dir = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/unattributed/{{concept}}",
        config = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/attribute_configs/{{attribute_config}}.yaml",
        spl_labels = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/filtered_sample_ids_and_labels.csv",
    output:
        directory(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/attributed/{{concept}}_{{attribute_config}}/{{fold}}/")
    resources:
        cores=1,
        mem="6G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/attribute_concept/{{concept}}_{{attribute_config}}/{{fold}}"
    shell:
        "5_attribute/scripts/attribute_concept.py {input} {output}"
