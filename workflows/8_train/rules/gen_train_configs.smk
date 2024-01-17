# Generates a zoo of configs based on a base cofig
rule gen_train_configs:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/base_configs/train_models_base_config.yaml"
    params:
        path_to_output_dir=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/train_configs/models"
    resources:
        cores=1,
        mem="1G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/gen_train_configs/gen_train_configs"
    shell:
        "scripts/gen_configs.py {input} {params.path_to_output_dir}"
