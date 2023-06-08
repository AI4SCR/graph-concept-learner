rule standard_normalize:
    input:
        f"{root}/intermediate_data/so.pkl",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/samples_splits.csv",
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/so_norm.pkl"
    params:
        cofactor = 5
    resources:
        cores=1,
        mem="4g",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/standard_normalize/standard_normalize"
    shell:
        "3_normalize/scripts/standard_normalize.py {input} {params} {output}"
