# Split samples into train test and validations sets, mantainign class proportions.
rule randomize_labels:
    input:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/samples_splits.csv",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/filtered_sample_ids_and_labels.csv"
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/randomized_data/samples_splits.csv",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/randomized_data/filtered_sample_ids_and_labels.csv"
    params:
        prediction_target=prediction_target,
        split_how=split_how
    resources:
        cores=1,
        mem="1g",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/randomize_labels/randomize_labels"
    shell:
        "80_rand_sanity_check/scripts/randomize_labels.py {input} {params.prediction_target} {params.split_how} {output}"
