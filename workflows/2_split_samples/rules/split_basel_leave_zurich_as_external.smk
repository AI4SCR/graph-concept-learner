# Split samples into train test and validations sets, mantainign class proportions.
rule split_basel_leave_zurich_as_external:
    input:
        f"{root}/intermediate_data/so.pkl",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/filtered_sample_ids_and_labels.csv",
    output:
        directory(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/CV_folds/folds"),
        directory(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/CV_folds/proportions"),
    params:
        split_proportions=[0.7, 0.15, 0.15], # Approximate proportions of the train test and validation splits (respectively).
        n_folds=config["n_folds"]
    resources:
        cores=2,
        mem="3G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/split_basel_leave_zurich_as_external/split_basel_leave_zurich_as_external"
    shell:
        "2_split_samples/scripts/split_basel_leave_zurich_as_external.py "
        "{input} "
        "{config[prediction_target]} {params.split_proportions} {params.n_folds} "
        "{output}"
