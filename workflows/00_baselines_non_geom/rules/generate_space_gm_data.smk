rule generate_space_gm_data:
    input:
        so_file = f"{root}/intermediate_data/so.pkl",
        cell_type_csv_path=f"{root}/raw_data/unzipped/Cluster_labels/Metacluster_annotations.csv",
        filered_samples=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/filtered_sample_ids_and_labels.csv"
    output:
        path_to_count=f"{root}/intermediate_data/non_spatial_baseline/count_dataset/composition_vectors.csv",
        path_to_norm=f"{root}/intermediate_data/non_spatial_baseline/frequency_dataset/composition_vectors.csv",
        path_to_count_ERless=f"{root}/intermediate_data/non_spatial_baseline/count_dataset_ERless/composition_vectors.csv",
        path_to_norm_ERless=f"{root}/intermediate_data/non_spatial_baseline/frequency_dataset_ERless/composition_vectors.csv"
    resources:
        cores=1,
        mem="2G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/logs/generate_space_gm_data/generate_space_gm_data"
    shell:
        "00_baselines_non_geom/scripts/generate_space_gm_data.py "
        "{input.so_file} {input.cell_type_csv_path} {input.filered_samples} "
        "{output.path_to_count} {output.path_to_norm} {output.path_to_count_ERless} {output.path_to_norm_ERless}"
