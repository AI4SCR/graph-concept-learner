# Generates a zoo of configs based on a base cofig
rule gen_train_configs:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/base_configs/train_gcl_base_config.yaml"
    output:
        protected(directory(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/train_model_configs"))
    resources:
        cores=1,
        mem="1G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/gen_train_configs/gen_train_configs"
    shell:
        "scripts/gen_configs.py {input} {output}"
