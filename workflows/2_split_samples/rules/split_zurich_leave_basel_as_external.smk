# Split samples into train test and validations sets, mantainign class proportions.
rule split_zurich_leave_basel_as_external:
    input:
        f"{root}/intermediate_data/so.pkl",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/filtered_sample_ids_and_labels.csv",
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/samples_splits.csv",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/samples_splits_proportions.csv"
    params:
        split_proportions=[0.7, 0.15, 0.15] # Approximate proportions of the train test and validation splits (respectively).
    resources:
        cores=2,
        mem="3g",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/split_zurich_leave_basel_as_external/split_zurich_leave_basel_as_external"
    shell:
        "2_split_samples/scripts/split_zurich_leave_basel_as_external.py "
        "{input} "
        "{config[prediction_target]} {params.split_proportions} "
        "{output}"
