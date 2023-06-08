# Filters out samples without prediction label or too few cells
rule filter_samples:
    input:
        #cfg_files=get_paths_to_dataset_configs,
        cfg_dir=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/dataset_configs/",
        so=f"{root}/intermediate_data/so.pkl"
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/filtered_sample_ids_and_labels.csv",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/label_dict.yaml"
    params:
        min_num_cells_per_type=10 # All concept graphs with less than this number of cells will be excluded.
    resources:
        cores=2,
        mem="3g",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/filter_samples/filter_samples"
    shell:
        "1_filter_samples/scripts/filter_samples.py "
        "{input.cfg_dir} {input.so} "
        "{config[prediction_target]} {params.min_num_cells_per_type} "
        "{output}"
