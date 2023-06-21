# Unzip raw data form the Jackson papaer
rule unzip_raw_data:
    input:
        f"{root}/raw_data/zipped/TumorStroma_masks.zip",
        f"{root}/raw_data/zipped/singlecell_locations.zip",
        f"{root}/raw_data/zipped/singlecell_cluster_labels.zip",
        f"{root}/raw_data/zipped/SingleCell_and_Metadata.zip",
        f"{root}/raw_data/zipped/OMEandSingleCellMasks.zip",
        raw_data_dir=f"{root}/raw_data"
    output:
        f"{root}/raw_data/unzipped/Data_publication/BaselTMA/Basel_PatientMetadata.csv",
        f"{root}/raw_data/unzipped/Data_publication/ZurichTMA/Zuri_PatientMetadata.csv",
        f"{root}/raw_data/unzipped/Data_publication/BaselTMA/SC_dat.csv",
        f"{root}/raw_data/unzipped/Data_publication/ZurichTMA/SC_dat.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/PG_zurich.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/PG_basel.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/Zurich_matched_metaclusters.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/Basel_metaclusters.csv",
        f"{root}/raw_data/unzipped/singlecell_locations/Zurich_SC_locations.csv",
        f"{root}/raw_data/unzipped/singlecell_locations/Basel_SC_locations.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/Metacluster_annotations.csv",
        f"{root}/raw_data/unzipped/Data_publication/Basel_Zuri_StainingPanel.csv",
        directory(f"{root}/raw_data/unzipped/OMEnMasks/Basel_Zuri_masks")
    resources:
        cores=10,
        mem="1G",
        queue="x86_1h",
    log:
        f"{root}/logs/unzip_raw_data/unzip_raw_data"
    shell:
        "0_make_so_jakson/scripts/unzip_raw_data.sh {input.raw_data_dir}"
