# Gather the common metadata between the two cohorts and write it to a df
rule gather_spl_metadata:
    input:
        f"{root}/raw_data/unzipped/Data_publication/BaselTMA/Basel_PatientMetadata.csv",
        f"{root}/raw_data/unzipped/Data_publication/ZurichTMA/Zuri_PatientMetadata.csv"
    output:
        f"{root}/intermediate_data/spl_meta_data.csv"
    resources:
        cores=2,
        mem="1g",
        queue="x86_1h",
    log:
        f"{root}/logs/gather_spl_metadata/gather_spl_metadata"
    shell:
        "0_make_so_jakson/scripts/gather_spl_metadata.py {input} {output}"
