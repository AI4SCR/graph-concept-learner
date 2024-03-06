rule generate_space_gm_data:
    input:
        so_file = f"{root}/intermediate_data/so.pkl",
        cell_type_csv_path=f"{root}/raw_data/unzipped/Cluster_labels/Metacluster_annotations.csv",
        filered_samples=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/CV_folds/folds/{{fold}}.csv"
    output:
        path_to_count=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/spage_gm_baseline_normalized_data/ER/{{fold}}.csv",
        path_to_count_ERless=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/spage_gm_baseline_normalized_data/ERless/{{fold}}.csv",
    resources:
        cores=1,
        mem="2G",
        queue="x86_1h",
    log:
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/logs/generate_space_gm_data/{{fold}}"
    shell:
        "00_baselines_non_geom/scripts/generate_space_gm_data.py "
        "{input.so_file} {input.cell_type_csv_path} {input.filered_samples} "
        "{output.path_to_count} {output.path_to_count_ERless}"
