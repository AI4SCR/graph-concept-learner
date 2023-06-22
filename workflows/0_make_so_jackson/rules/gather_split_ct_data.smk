# Gather and split single cell data regarding cell type and location
rule gather_split_ct_data:
    input:
        f"{root}/raw_data/unzipped/Cluster_labels/PG_zurich.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/PG_basel.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/Zurich_matched_metaclusters.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/Basel_metaclusters.csv",
        f"{root}/raw_data/unzipped/singlecell_locations/Zurich_SC_locations.csv",
        f"{root}/raw_data/unzipped/singlecell_locations/Basel_SC_locations.csv",
        f"{root}/raw_data/unzipped/Cluster_labels/Metacluster_annotations.csv"
    output:
        directory(f"{root}/intermediate_data/sct_data")
    resources:
        cores=2,
        mem="1G",
        queue="x86_1h",
    log:
        f"{root}/logs/gather_split_ct_data/gather_split_ct_data"
    shell:
        "0_make_so_jackson/scripts/gather_split_ct_data.sh {input} {output}"
