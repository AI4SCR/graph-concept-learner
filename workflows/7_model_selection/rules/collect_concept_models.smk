# Collect the models into one big config
rule collect_concept_models:
    input:
        expand(
            f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_model_per_concept/"+"{{labels_permuted}}/{{attribute_config}}/{concept}.yaml",
            concept=[os.path.splitext(f)[0] for f in os.listdir(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs") if f.endswith(".yaml")]
        )
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/train_configs/concept_sets/{{labels_permuted}}/{{attribute_config}}/concept_set.yaml"
    resources:
        cores="1",
        mem="1G",
        queue="x86_1h",
    params:
        exclude=[

        ]
    log:
       f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/collect_concept_models/{{labels_permuted}}/{{attribute_config}}"
    shell:
        "7_model_selection/scripts/collect_concept_models.py "
        "$(dirname $(echo {input} | cut -d' ' -f1)) "
        "{output} "
        "{params.exclude}"
