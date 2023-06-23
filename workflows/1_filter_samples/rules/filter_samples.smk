# Filters out samples without prediction label or too few cells
rule filter_samples:
    input:
        cfg_files=get_paths_to_dataset_configs,
        #cfg_dir=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/dataset_configs/",
        so=f"{root}/intermediate_data/so.pkl"
    output:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/filtered_sample_ids_and_labels.csv",
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/label_dict.yaml"
    params:
        min_num_cells_per_graph=10 # All concept graphs with less than this number of nodes will be excluded.
    resources:
        cores=2,
        mem="3G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/filter_samples/filter_samples"
    shell:
        "1_filter_samples/scripts/filter_samples.py "
        "$(dirname {input.cfg_files[0]}) {input.so} "
        "{config[prediction_target]} {params.min_num_cells_per_graph} "
        "{output}"
