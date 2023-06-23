# Takes the data split from one Cv fold and normalizes each split separately.
# The normalized data is then stores in pkl file, which is a dictionary of all the
# normalized data.
rule min_max_normalize:
    input:
        f"{root}/intermediate_data/so.pkl",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/CV_folds/folds/{{fold}}.csv",
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/normalized_data/{{fold}}.pkl"
    params:
        cofactor = 5
    resources:
        cores=1,
        mem="4G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/min_max_normalize/{{fold}}"
    shell:
        "3_normalize/scripts/min_max_normalize.py {input} {params} {output}"
