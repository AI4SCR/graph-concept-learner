# Generates a zoo of configs based on a base cofig
rule gen_pretrain_configs:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/base_configs/pretrain_models_base_config.yaml"
    output:
        directory(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/pretrain_model_configs")
    resources:
        cores=1,
        mem="1G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/gen_pretrain_configs/gen_pretrain_configs"
    shell:
        "scripts/gen_configs.py {input} {output}"
