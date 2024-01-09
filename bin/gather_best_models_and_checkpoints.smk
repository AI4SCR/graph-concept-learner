rule gather_best_models_and_checkpoints:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_model_per_concept/*.yaml"
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_combinations/all_concepts.yaml"
    resources:
        cores="1",
        mem="1G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/gather_best_models_and_checkpoints/gather_best_models_and_checkpoints"
    shell:
        "for f in $(realpath $(dirname {input[0]})/*); do "
        "cat $f >> {output}; "
        "done"
