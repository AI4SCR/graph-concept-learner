# Split single cell channel data into per sample csv's
rule split_scc_data:
    input:
        f"{root}/raw_data/unzipped/Data_publication/BaselTMA/SC_dat.csv",
        f"{root}/raw_data/unzipped/Data_publication/ZurichTMA/SC_dat.csv"
    output:
        directory(f"{root}/intermediate_data/scc_data")
    resources:
        cores=10,
        mem="10G",
        queue="x86_1h",
    log:
        f"{root}/logs/split_scc_data/split_scc_data"
    shell:
        "0_make_so_jackson/scripts/split_scc_data.sh {input} {output}"
