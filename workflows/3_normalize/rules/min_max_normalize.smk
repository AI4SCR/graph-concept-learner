rule min_max_normalize:
    input:
        f"{root}/intermediate_data/so.pkl",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/samples_splits.csv",
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/so_norm.pkl"
    params:
        cofactor = 5
    resources:
        cores=1,
        mem="4G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/min_max_normalize/min_max_normalize"
    shell:
        "3_normalize/scripts/min_max_normalize.py {input} {params} {output}"
